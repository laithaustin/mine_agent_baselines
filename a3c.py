import minerl
import torch as th
import torch.nn as nn
import multiprocessing as mp
import tqdm
import threading
import gym
import logging
logging.basicConfig(level=logging.DEBUG)

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
        self.device = "cpu"
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
        in_channels = self.env.observation_space.shape[0]
        out_channels = 16
        num_actions = self.env.action_space.n
        self.global_value = ValueCNN(in_channels, out_channels, num_actions)
        self.global_policy = PolicyCNN(in_channels, out_channels, num_actions)
        # share memory between processes
        self.global_value.share_memory()
        self.global_policy.share_memory()
        # define optimizer using RMSProp with shared memory
        self.optimizer_val = th.optim.RMSprop(self.global_value.parameters(), lr=0.0001)
        self.optimizer_policy = th.optim.RMSprop(self.global_policy.parameters(), lr=0.0001)

    def train(self):
        """
        Train the model using the A3C algorithm. Should also output graphs and the 
        total reward gathered by the agent across multiple episodes.
        """
        # number of processes
        num_processes = mp.cpu_count() if self.num_workers == -1 else self.num_workers

        # define workers
        workers = []
        for _ in range(1):
            worker = A3C_Worker(self, _)
            workers.append(worker)

        # start workers
        for worker in workers:
            worker.start()

        # join workers
        for worker in workers:
            worker.join()

        # save model
        th.save(self.global_policy.state_dict(), "policy_model.pth")
        th.save(self.global_value.state_dict(), "value_model.pth")

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
        self.local_value = ValueCNN(glbl.env.observation_space.shape[0], 16, glbl.env.action_space.n)
        self.local_policy = PolicyCNN(glbl.env.observation_space.shape[0], 16, glbl.env.action_space.n)
        self.worker_id = worker_id
        self.env = glbl.env
        self.device = glbl.device
        self.T_max = glbl.T_max
        self.glbl = glbl
        # define optimizers
        self.optim_val = glbl.optimizer_val
        self.optim_policy = glbl.optimizer_policy
        # set local counters
        self.t = 0
        self.t_max = 10

    def run(self):
        while self.glbl.T < self.glbl.T_max:
            # reset gradients of the global model
            self.optim_val.zero_grad()
            self.optim_policy.zero_grad()

            # reset gradients
            d_policy = 0
            d_value = 0

            # sync with global network
            self.local_policy.load_state_dict(self.glbl.global_policy.state_dict())
            self.local_value.load_state_dict(self.glbl.global_value.state_dict())

            # reset environment
            obs = self.env.reset()
            print(obs)
            done = False
            t_start = self.t

            traj = []
            while not done and not self.t - t_start == self.t_max:
                action, log_prob = self.local_policy(obs, return_prob=True)
                obs, reward, done, _ = self.env.step(action)
                self.t += 1
                self.glbl.T += 1
                traj.append((obs, action, reward, log_prob))
            
            R = 0 if done else self.local_value(obs)

            for i in range(len(traj) - 1, -1, -1):
                obs, action, reward, log_prob = traj[i]
                R = reward + self.glbl.gamma * R
                # calculate loss
                adv = R - self.local_value(obs)
                p_loss = -log_prob * adv
                v_loss = adv ** 2
                # accumulate gradients
                d_policy += th.autograd.grad(p_loss, self.local_policy.parameters())
                d_value += th.autograd.grad(v_loss, self.local_value.parameters())
                # updat local network
                self.optim_val.step()
                self.optim_policy.step()
                print(f"Worker {self.worker_id}, episode {self.glbl.T}, policy_loss: {p_loss}, value_loss: {v_loss}")
            
            with self.glbl.lock:
                # update global network with mutex lock
                for p, gp in zip(self.glbl.global_policy.parameters(), d_policy):
                    p.grad = gp if p.grad is None else p.grad + gp
                for p, gv in zip(self.glbl.global_value.parameters(), d_value):
                    p.grad = gv if p.grad is None else p.grad + gv

                self.glbl.optimizer_policy.step()
                self.glbl.optimizer_val.step()

        self.glbl.env.close()

class ValueCNN(nn.Module):
    """
    Convolutional Neural Network that will be used by the Actor-Critic model. 
    Should have at least two convolutional layers and two fully connected layers.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size = 3, 
        stride = 1, 
        padding = 0
    ):
        """
        Give the model 2 convolutional layers and 2 fully connected layers.
        """
        super(ValueCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, input):
        """
        Forward pass through the model.
        """
        x = self.conv1(input)
        x = self.conv2(th.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(th.relu(x))
        x = self.fc2(th.relu(x))
        return x
    
class PolicyCNN(nn.Module):
    """
    Convolutional Neural Network that will be used by the Actor-Critic model. 
    Should have some CNN layers and output logits for the action space. Also
    should have a method that returns the action given a state.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        num_actions,
        kernel_size = 3, 
        stride = 1, 
        padding = 0
    ):
        super(PolicyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.fc1 = nn.Linear(out_channels, 128)
        self.fc2 = nn.Linear(128, num_actions)

        
    def forward(self, input, return_prob=False):
        """
        Forward pass through the model.
        """
        x = self.conv1(input)
        x = self.conv2(th.relu(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(th.relu(x))
        x = self.fc2(th.relu(x))
        if return_prob:
            # return both action and its log probability
            return th.argmax(th.softmax(x, dim=-1)).item(), th.log_softmax(x, dim=-1)
        else:
            return th.argmax(th.softmax(x, dim=-1)).item()


def test(env, policy_model_path, num_episodes=10):
    """
    Test the trained model on the environment. Should also record the agent's actions and the environment's state. Also should
    output graphs and the total reward gathered by the agent across multiple episodes.
    """
    # load model from path
    policy_model = PolicyCNN(env.observation_space.shape[0], 16, env.action_space.n)
    policy_model.load_state_dict(th.load(policy_model_path))
    policy_model.eval()

    env = gym.make('MineRLTreechop-v0').env

    for i in tqdm.tqdm(range(num_episodes)):
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        env.render()

        while not done:
            # get action from model
            action = policy_model(obs)

            # take action
            obs, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            steps += 1

        print(f"Episode {i}, total reward: {total_reward} in {steps} steps.")

if __name__ == "__main__":
    # train the model
    a3c = A3C_Orchestrator("MineRLTreechop-v0", 100, 4, 1000, 0.99, 0.0001)
    a3c.train()
    a3c.env.close()
    # test results on environment
    # test()
