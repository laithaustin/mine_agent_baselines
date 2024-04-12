import minerl
import torch as th
import torch.nn as nn
import multiprocessing as mp
import tqdm
import threading
import gym
import logging
import numpy as np
# logging.basicConfig(level=logging.DEBUG)
from torchsummary import summary
th.autograd.set_detect_anomaly(True)

# reduce the amount of observations - code via rl_plus_script.py
class PovOnlyObservation(gym.ObservationWrapper):
    """
    Turns the observation space into POV only, ignoring the inventory. This is needed for stable_baselines3 RL agents,
    as they don't yet support dict observations. The support should be coming soon (as of April 2021).
    See following PR for details:
    https://github.com/DLR-RM/stable-baselines3/pull/243
    """
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']
# reduce action space - code via rl_plus_script.py  
class ActionShaping(gym.ActionWrapper):
    """
    The default MineRL action space is the following dict:

    Dict(attack:Discrete(2),
         back:Discrete(2),
         camera:Box(low=-180.0, high=180.0, shape=(2,)),
         craft:Enum(crafting_table,none,planks,stick,torch),
         equip:Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         forward:Discrete(2),
         jump:Discrete(2),
         left:Discrete(2),
         nearbyCraft:Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe),
         nearbySmelt:Enum(coal,iron_ingot,none),
         place:Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch),
         right:Discrete(2),
         sneak:Discrete(2),
         sprint:Discrete(2))

    It can be viewed as:
         - buttons, like attack, back, forward, sprint that are either pressed or not.
         - mouse, i.e. the continuous camera action in degrees. The two values are pitch (up/down), where up is
           negative, down is positive, and yaw (left/right), where left is negative, right is positive.
         - craft/equip/place actions for items specified above.
    So an example action could be sprint + forward + jump + attack + turn camera, all in one action.

    This wrapper makes the action space much smaller by selecting a few common actions and making the camera actions
    discrete. You can change these actions by changing self._actions below. That should just work with the RL agent,
    but would require some further tinkering below with the BC one.
    """
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)

        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            # [('back', 1)],
            # [('left', 1)],
            # [('right', 1)],
            # [('jump', 1)],
            # [('forward', 1), ('attack', 1)],
            # [('craft', 'planks')],
            [('forward', 1), ('jump', 1)],
            [('camera', [-self.camera_angle, 0])],
            [('camera', [self.camera_angle, 0])],
            [('camera', [0, self.camera_angle])],
            [('camera', [0, -self.camera_angle])],
        ]

        self.actions = []
        for actions in self._actions:
            act = self.env.action_space.noop()
            for a, v in actions:
                act[a] = v
            if self.always_attack:
                act['attack'] = 1
            self.actions.append(act)

        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action]

class A3C_Orchestrator():
    """
    Orchestrator class that will manage the training of the A3C model. 
    Should also manage the workers and the global network.
    """

    def __init__(
        self, 
        env_name, 
        num_episodes, 
        num_workers, 
        T_max, 
        gamma, 
        lr
    ):
        # define shared counter
        self.T = 0
        # define device
        self.device = "mps"
        # define environment
        print(f"Creating environment {env_name}")
        self.env = gym.make(env_name)
        self.env = PovOnlyObservation(self.env)
        self.env = ActionShaping(self.env)
        self.num_episodes = num_episodes
        self.num_workers = num_workers
        self.T_max = T_max
        self.gamma = gamma
        self.lr = lr
        # create lock for shared memory
        self.lock = threading.Lock()
        # setup models
        self._setup_models()

    def _setup_models(self):
        """
        Setup the global and local models for the A3C algorithm.
        """
        print(f"observation space: {self.env.observation_space.shape}, action space: {self.env.action_space.n}")
        in_channels = self.env.observation_space.shape[-1]
        out_channels = 16
        num_actions = self.env.action_space.n
        height = self.env.observation_space.shape[0]
        width = self.env.observation_space.shape[1]
        self.global_net = ActorCritic(in_channels, out_channels, height, width, num_actions, 0, self.device).to(self.device)
        # define optimizer using RMSProp with shared memory
        self.optimizer = th.optim.RMSprop(self.global_net.parameters(), lr=self.lr)

    def train(self):
        """
        Train the model using the A3C algorithm. Should also output graphs and the 
        total reward gathered by the agent across multiple episodes.
        """
        # number of processes
        num_processes = mp.cpu_count() if self.num_workers == -1 else self.num_workers

        # define workers
        workers = []
        for _ in range(num_processes):
            worker = A3C_Worker(self, _)
            workers.append(worker)

        # start workers
        for worker in workers:
            worker.start()

        # join workers
        for worker in workers:
            worker.join()

        # save model
        th.save(self.global_net.state_dict(), "a3c_model.pth")

class A3C_Worker(threading.Thread):
    """
    Each worker will have its own instance of environment and accumulate 
    gradients that will update the global network.
    """
    def __init__(self,
                glbl: A3C_Orchestrator,
                worker_id,
        ):
        super().__init__()
        # initialize local network
        self.local_net = ActorCritic(glbl.env.observation_space.shape[-1], 16, glbl.env.observation_space.shape[0], glbl.env.observation_space.shape[1], glbl.env.action_space.n, 0, glbl.device).to(glbl.device)
        self.worker_id = worker_id
        self.env = glbl.env
        self.device = glbl.device
        self.T_max = glbl.T_max
        self.glbl = glbl
        # define optimizers
        self.optim_local = th.optim.RMSprop(self.local_net.parameters(), lr=glbl.lr)
        # set local counters
        self.t = 0
        self.t_max = 100

    def run(self):
        done = False
        obs = self.env.reset()
        episode = 0
        while self.glbl.T < self.glbl.T_max and episode < self.glbl.num_episodes:
            # reset gradients of the global model
            self.optim_local.zero_grad()
            self.glbl.optimizer.zero_grad()

            # reset gradients
            d_policy = []
            d_value = []

            # sync with global network
            self.local_net.load_state_dict(self.glbl.global_net.state_dict())
            # set to training mode
            self.local_net.train()

            t_start = self.t

            traj = []
            while not done and not self.t - t_start == self.t_max:
                action, value, log_prob = self.local_net(obs)
                obs, reward, done, _ = self.env.step(action)
                # break out of loop if done
                if done:
                    print("***************episode done***************")
                    break
                self.t += 1
                self.glbl.T += 1
                traj.append((obs, action, reward))
            
            with th.no_grad():
                R = 0 if done else self.local_net(obs)[1]

            for i in range(len(traj) - 1, -1, -1):
                # reset gradients
                self.optim_local.zero_grad()
                obs, action, reward = traj[i]
                # move to device
                obs = th.from_numpy(obs.copy()).float().to(self.device)
                R = reward + self.glbl.gamma * R
                action, value, log_prob = self.local_net(obs)
                # move to device
                log_prob = log_prob.to(self.device)
                value = value.to(self.device)
                # calculate loss
                adv = R - value
                p_loss = - log_prob * adv
                v_loss = adv ** 2
                # accumulate gradients
                loss = p_loss + v_loss
                loss.backward()
                # clip gradients
                th.nn.utils.clip_grad_norm_(self.local_net.parameters(), 0.5) # 0.5 is the max norm
                # update local network
                self.optim_local.step()
                # accumulate gradients
                d_policy += (p.grad.clone() for p in self.local_net.policy.parameters())
                d_value += (v.grad.clone() for v in self.local_net.value.parameters())
                
            print(f"episode {episode}, ts {self.t}, p_loss: {p_loss}, v_loss: {v_loss}")
            
            # reset done flag
            if done:
                done = False
                obs = self.env.reset()
                episode += 1
            
            if self.t % self.t_max == 0:
                # normalize gradients
                with self.glbl.lock:
                    # update global network with mutex lock
                    for p, g in zip(self.glbl.global_net.policy.parameters(), d_policy):
                        p.grad = g / self.t_max if p.grad is None else p.grad + g / self.t_max
                    for v, g in zip(self.glbl.global_net.value.parameters(), d_value):
                        v.grad = g if v.grad is None else v.grad + g / self.t_max
                    # update global network
                    self.glbl.optimizer.step()

        self.glbl.env.close()

class ActorCritic(nn.Module):
    """
    Model that ouputs value, policy and log probability of the action.
    Networks will share CNN layers and have separate fully connected layers.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels,
        height,
        width, 
        num_actions, # shape of the action space,  
        feature_dim, # shape of additional features to include in the model
        device="cpu"
    ):
        super(ActorCritic, self).__init__()
        self.device = device
        # process image data
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # compute the shape of the flattened output
        height = (height - 4) // 2 
        width = (width - 4) // 2 
        input_shape = out_channels * height * width + feature_dim
        # process additional features + flatten cnn output
        self.fc = nn.Linear(input_shape, 128)
        # separate fully connected layers
        self.policy = nn.Linear(128, num_actions)
        self.value = nn.Linear(128, 1)

    def forward(
        self, 
        x, 
        features=None # additional domain-specific features
    ):
        # check if input is a tensor
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device) # need to add batch dimension
        else:
            x = x.float().unsqueeze(0).to(self.device)
        # reshape so that channels are first
        x = x.permute(0, 3, 1, 2)

        # pass through CNN layers
        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = self.pool(x)
        x = self.flatten(x)
        if features is not None:
            x = th.cat([x, features], dim=1)
        x = th.relu(self.fc(x))

        # return logits for policy, action, and value
        probs = th.softmax(self.policy(x), dim=-1)
        action = th.multinomial(probs, num_samples=1).item()
        value = self.value(x)
        return action, value, th.log(probs[0, action])

def test(env_name, global_net, num_episodes=10):
    """
    Test the trained model on the environment. Should also record the agent's actions and the environment's state. Also should
    output graphs and the total reward gathered by the agent across multiple episodes.
    """
    # create environment
    env = gym.make(env_name)
    env = PovOnlyObservation(env)
    env = ActionShaping(env)

    global_net.eval()

    for i in range(num_episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        # env.render()

        while not done:
            # get action from model
            action, _, _ = global_net(obs)

            # take action
            obs, reward, done, _ = env.step(action)
            # env.render()
            total_reward += reward
            steps += 1

        print(f"Episode {i}, total reward: {total_reward} in {steps} steps.")

    env.close()

if __name__ == "__main__":
    #chopping tree
    task = "MineRLTreechop-v0"
    # train the model
    a3c = A3C_Orchestrator(task, 10, 1, 200000, 0.99, 0.0001)
    a3c.train()
    a3c.env.close()
    # # test results on environment
    # a3c = A3C_Orchestrator(task, 10, 1, 2000000, 0.99, 0.0001)
    # # load weights
    # a3c.global_net.load_state_dict(th.load("a3c_model.pth"))
    # test(task, a3c.global_net)
