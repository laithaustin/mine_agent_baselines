import minerl
import torch as th
import torch.nn as nn
import gym
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from models.Actor import Actor
from models.Critic import Critic
import wandb

DATA_DIR = "/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines"
BATCH_SIZE = 5

# Initialize wandb
wandb.init(project="minerl-a2c", config={
    "task": "MineRLTreechop-v0",
    "max_timesteps": 2000000,
    "gamma": 0.99,
    "learning_rate": 0.0001,
    "experiment_name": "a2c_bc_always_attack",
    "timesteps": 3600,
    "load_model": True,
    "entropy_start": 0.5,
})

# Observation Wrapper
class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']

# Action Wrapper
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


def dataset_action_batch_to_actions(dataset_actions, camera_margin=5):
    """
    Turn a batch of actions from dataset (`batch_iter`) to a numpy
    array that corresponds to batch of actions of ActionShaping wrapper (_actions).

    Camera margin sets the threshold what is considered "moving camera".

    Note: Hardcoded to work for actions in ActionShaping._actions, with "intuitive"
        ordering of actions.
        If you change ActionShaping._actions, remember to change this!

    Array elements are integers corresponding to actions, or "-1"
    for actions that did not have any corresponding discrete match.
    """
    # There are dummy dimensions of shape one
    camera_actions = dataset_actions["camera"].squeeze()
    attack_actions = dataset_actions["attack"].squeeze()
    forward_actions = dataset_actions["forward"].squeeze()
    jump_actions = dataset_actions["jump"].squeeze()
    batch_size = len(camera_actions)
    actions = np.zeros((batch_size,), dtype=np.int)

    for i in range(len(camera_actions)):
        # Moving camera is most important (horizontal first)
        if camera_actions[i][0] < -camera_margin:
            actions[i] = 3
        elif camera_actions[i][0] > camera_margin:
            actions[i] = 4
        elif camera_actions[i][1] > camera_margin:
            actions[i] = 5
        elif camera_actions[i][1] < -camera_margin:
            actions[i] = 6
        elif forward_actions[i] == 1:
            if jump_actions[i] == 1:
                actions[i] = 2
            else:
                actions[i] = 1
        elif attack_actions[i] == 1:
            actions[i] = 0
        else:
            # No reasonable mapping (would be no-op)
            actions[i] = -1
    return actions

def compute_gae(next_value, rewards, masks, values, gamma=1.0, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def get_entropy_linear(start, end, current, total_steps):
    """
    Linear entropy annealing function.
    """
    if current >= total_steps:
        return end
    return start + (end - start) * current / total_steps

# Training Loop
def train_a2c(
        max_timesteps, 
        gamma, 
        lr,
        env,
        experiment_name="a2c",
        timesteps=3600,
        load_model=False,
        entropy_start=0.0 # starting entropy value,
    ):

    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    critic = Critic(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], device).to(device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    
    if load_model:
        actor.load_state_dict(th.load("a2c_bc_model.pth"))

    actor_optimizer = th.optim.RMSprop(actor.parameters(), lr=lr)
    critic_optimizer = th.optim.RMSprop(critic.parameters(), lr=lr)
    
    global_step = 0
    mean_rewards = []
    local_rewards_arr = []
    global_reward = 0
    episode = 0
    time_start = time.time()
    while global_step < max_timesteps:
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        rewards = []
        states = []
        masks = []
        values = []
        log_probs = []
        actions = []


        while not done:
            # normalize observations and permute shape
            obs = obs.astype(np.float32) / 255.0
            state = th.tensor(obs.copy(), dtype=th.float32).to(device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            
            with th.no_grad():
                action, log_prob, _ = actor(state)
                value = critic(state)
            next_obs, reward, done, _ = env.step(action)

            rewards.append(reward)
            masks.append(1.0 - done)
            values.append(value.item())
            log_probs.append(log_prob)
            states.append(state)
            actions.append(action)

            obs = next_obs
            total_reward += reward
            global_step += 1
            steps += 1

            if steps >= timesteps:
                done = True

            # log rewards
            wandb.log({"reward": reward, "total_reward": total_reward, "steps": steps, "global_step": global_step})
            
            # update actor and critic every 10 steps
            if steps % 10 == 0 or done:
                with th.no_grad():
                    next_value = critic(state) if not done else 0
                returns = compute_gae(next_value, rewards, masks, values, gamma)        

                values = critic(th.stack(states).to(device).squeeze())
                log_probs = actor(th.stack(states).to(device).squeeze())[1]
                returns = th.tensor(returns, dtype=th.float32).to(device)
                advantages = returns - values

                # add entropy to loss
                if entropy_start > 0:
                    entropy_coef = get_entropy_linear(entropy_start, 0.0, global_step, max_timesteps)
                else:
                    entropy_coef = 0.0
                entropy = -th.mean(-log_probs)

                # update actor
                actor_optimizer.zero_grad()
                actor_loss = -(advantages.detach() * log_probs).mean() + entropy_coef * entropy
                actor_loss.backward()
                actor_optimizer.step()

                # update critic
                critic_optimizer.zero_grad()
                critic_loss = advantages.pow(2).mean() + entropy_coef * entropy.detach()
                critic_loss.backward()
                critic_optimizer.step()

                if steps % 1000 == 0:
                    print(f"Ep: {episode}, TS: {steps}, A_Loss: {actor_loss}, C_Loss: {critic_loss}")
                    wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss, "steps": steps, "global_step": global_step})
                
                # reset variables
                rewards = []
                states = []
                masks = []
                values = []
                log_probs = []
                actions = []

            if done:
                break

        episode += 1
        print(f"Episode {episode}, total reward: {total_reward}")
        global_reward += total_reward
        mean_rewards.append(total_reward / steps)
        local_rewards_arr.append(total_reward)

        wandb.log({"episode": episode, "total_reward": total_reward, "steps": steps, "global_step": global_step})
        wandb.log({"mean_reward": total_reward / steps, "global_reward": global_reward})

        # checkpoint models
        if episode % 100 == 0:
            th.save(actor.state_dict(), f"{experiment_name}_actor_{episode}.pth")
            th.save(critic.state_dict(), f"{experiment_name}_critic_{episode}.pth")


    print("*********************************************")
    print(f"Training took {time.time() - time_start} seconds")
    print(f"Mean reward per episode: {global_reward / episode}")
    print(f"Total episodes: {episode}")
    print("Total rewards during training: ", global_reward)

    # Save graph of mean rewards over episodes
    plt.plot(mean_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Rewards")
    plt.title("Mean Rewards vs Episodes")
    plt.savefig("mean_rewards.png")

    # Save graph of local rewards over episodes
    plt.plot(local_rewards_arr)
    plt.xlabel("Episodes")
    plt.ylabel("Local Rewards")
    plt.title("Local Rewards vs Episodes")
    plt.savefig("local_rewards.png")
    env.close()

    # Save models seperately
    th.save(actor.state_dict(), f"{experiment_name}_actor.pth")
    th.save(critic.state_dict(), f"{experiment_name}_critic.pth")

def test_a2c(env_name, episodes, load_model=False):
    env = gym.make(env_name)
    env = PovOnlyObservation(env)
    env = ActionShaping(env)
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    critic = Critic(env.observation_space.shape[-1], 64, env.observation_space.shape[0], env.observation_space.shape[1], device).to(device)
    if load_model:
        checkpoint = th.load("model.pth")
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
    actor.eval()
    critic.eval()

    global_reward = 0
    for episode in range(episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            action, _, _ = actor(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        global_reward += total_reward
        print(f"Episode {episode}, total reward: {total_reward}")

    print("*********************************************")
    print(f"Mean reward per episode: {global_reward / episodes}")
    env.close()

# pretrain actor with behavioral cloning
def train_a2c_w_bc(
    dir, 
    task, 
    epochs, 
    learning_rate):
    """
    Use behavioral cloning to pretrain the actor network before training with A2C.
    """

    # train actor network with behavioral cloning
    data = minerl.data.make(task,  data_dir=dir, num_workers=4)
    device = th.device("cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu")
    # We know ActionShaping has seven discrete actions, so we create
    # a network to map images to seven values (logits), which represent
    # likelihoods of selecting those actions
    network = Actor(3, 64, 64, 7, device, True).to(device)
    optimizer = th.optim.Adam(network.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    iter_count = 0
    losses = []
    for dataset_obs, dataset_actions, _, _, _ in tqdm(data.batch_iter(num_epochs=epochs, batch_size=32, seq_len=1)):
        # We only use pov observations (also remove dummy dimensions)
        obs = dataset_obs["pov"].squeeze().astype(np.float32)
        # Transpose observations to be channel-first (BCHW instead of BHWC)
        obs = obs.transpose(0, 3, 1, 2)
        # Normalize observations
        obs /= 255.0

        # Actions need bit more work
        actions = dataset_action_batch_to_actions(dataset_actions)

        # Remove samples that had no corresponding action
        mask = actions != -1
        obs = obs[mask]
        actions = actions[mask]

        # Obtain logits of each action
        logits = network(th.from_numpy(obs).to(device))[-1]

        # Minimize cross-entropy with target labels.
        # We could also compute the probability of demonstration actions and
        # maximize them.
        loss = loss_function(logits, th.from_numpy(actions).long().to(device))

        # Standard PyTorch update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_count += 1
        losses.append(loss.item())
        if (iter_count % 1000) == 0:
            mean_loss = sum(losses) / len(losses)
            tqdm.write("Iteration {}. Loss {:<10.3f}".format(iter_count, mean_loss))
            losses.clear()

    th.save(network.state_dict(), "a2c_bc_model.pth")
    del data
    del network


def test_a2c_w_bc():
    """
    TODO: complete this function
    """
    pass

def make_env(task, camera_angle=10, always_attack=False, simple_test=False):
    if simple_test:
        # test algo on cartpole
        env = gym.make("CartPole-v1")
        env = gym.wrappers.TransformReward(env, lambda r: r / 100.0)
        env = gym.wrappers.FlattenObservation(env)
        return env
    env = gym.make(task)
    env = PovOnlyObservation(env)
    env = ActionShaping(env, camera_angle, always_attack)
    return env

if __name__ == "__main__":
    env = make_env("MineRLTreechop-v0", always_attack=True)
    # train_a2c_w_bc(DATA_DIR, task, 5, 0.0001)
    train_a2c(2000000, .99, 0.0001, env, 'a2c_bc', 3600, True, 0.5)


