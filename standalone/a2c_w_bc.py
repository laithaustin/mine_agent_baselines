import minerl
import torch as th
import torch.nn as nn
import gym
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

DATA_DIR = "/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines"

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

class Actor(nn.Module):
    def __init__(self, in_channels, height, width, num_actions, device="cpu", bc=False):
        super(Actor, self).__init__()
        self.device = device
        self.bc = bc
        
        # DQN Nature paper architecture
        self.cnn1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.flatten(self.cnn3(self.cnn2(self.cnn1(th.zeros(1, in_channels, height, width))))).shape[1]

        self.fc = nn.Linear(n_flatten, 512)  # Adjust the input features to match your input size
        self.policy = nn.Linear(512, num_actions)

    def forward(self, x):
        x = self.prepare_input(x)
        x = th.relu(self.fc(x))
        logits = self.policy(x)
        probs = th.softmax(logits, dim=-1)
        action = th.multinomial(probs, num_samples=1)
        log_prob = th.log(probs[0, action])
        return action, log_prob, probs

    def prepare_input(self, x):
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)
        if not self.bc:
            x = x.permute(0, 3, 1, 2)  # Assume input shape (Batch, Height, Width, Channels)
        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = th.relu(self.cnn3(x))
        x = self.flatten(x)
        return x

class Critic(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, device="cpu"):
        super(Critic, self).__init__()
        self.device = device
        # Shared feature extraction layers with the Actor
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        height = (height - 4) // 2 
        width = (width - 4) // 2 
        input_shape = out_channels * height * width
        self.fc = nn.Linear(input_shape, 128)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = self.prepare_input(x)
        x = th.relu(self.fc(x))
        value = self.value(x)
        return value

    def prepare_input(self, x):
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)
        x = x.permute(0, 3, 1, 2)
        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = self.pool(x)
        x = self.flatten(x)
        return x

# Training Loop
def train_a2c(env_name, max_timesteps, gamma, lr, timesteps=3600, load_model=False):
    env = gym.make(env_name)
    env = PovOnlyObservation(env)
    env = ActionShaping(env)
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    if load_model:
        actor.load_state_dict(th.load("a2c_bc_model.pth"))
    critic = Critic(env.observation_space.shape[-1], 64, env.observation_space.shape[0], env.observation_space.shape[1], device).to(device)
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
        local_rewards = 0
        while not done and steps < timesteps:
            action, log_prob, _ = actor(obs)
            value = critic(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward

            # Get the value for the next state
            with th.no_grad():
                next_value = critic(obs) if not done else 0

            # Calculate advantage and update models
            advantage = reward + gamma * next_value - value
            actor_loss = -log_prob * advantage.detach()
            critic_loss = advantage.pow(2)

            # Update actor
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # Update critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            steps += 1
            global_step += 1
            local_rewards += reward

        global_reward += total_reward
        mean_rewards.append(local_rewards / steps)
        local_rewards_arr.append(local_rewards)
        print(f"Episode {episode}, total reward: {total_reward}")
        episode += 1

    print("*********************************************")
    print(f"Training took {time.time() - time_start} seconds")
    print(f"Mean reward per episode: {global_reward / episode}")

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

    # Save model
    th.save({'actor_state_dict': actor.state_dict(), 'critic_state_dict': critic.state_dict()}, "model.pth")

def test_a2c():
    env = gym.make("MineRLTreechop-v0")
    env = PovOnlyObservation(env)
    env = ActionShaping(env)
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    model = ActorCritic(env.observation_space.shape[-1], 16, env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    model.load_state_dict(th.load("model.pth"))
    model.eval()
    obs = env.reset()
    done = False
    while not done:
        action, _, _ = model(obs)
        obs, _, done, _ = env.step(action)
        env.render()
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
    pass


if __name__ == "__main__":
    task = "MineRLTreechop-v0"
    train_a2c_w_bc(DATA_DIR, task, 5, 0.0001)
    train_a2c(task, 100000, 0.0001, 0.99, 3600, load_model=True)


