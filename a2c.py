import minerl
import torch as th
import torch.nn as nn
import gym
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import time

th.autograd.set_detect_anomaly(True)

# Observation Wrapper
class PovOnlyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space['pov']

    def observation(self, observation):
        return observation['pov']

# Action Wrapper
class ActionShaping(gym.ActionWrapper):
    def __init__(self, env, camera_angle=10, always_attack=False):
        super().__init__(env)
        self.camera_angle = camera_angle
        self.always_attack = always_attack
        self._actions = [
            [('attack', 1)],
            [('forward', 1)],
            [('back', 1)],
            [('left', 1)],
            [('right', 1)],
            [('jump', 1)],
            [('forward', 1), ('attack', 1)],
            [('craft', 'planks')],
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

class Actor(nn.Module):
    def __init__(self, in_channels, out_channels, height, width, num_actions, device="cpu"):
        super(Actor, self).__init__()
        self.device = device
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        height = (height - 4) // 2 
        width = (width - 4) // 2 
        input_shape = out_channels * height * width
        self.fc = nn.Linear(input_shape, 128)
        self.policy = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.prepare_input(x)
        x = th.relu(self.fc(x))
        logits = self.policy(x)
        probs = th.softmax(logits, dim=-1)
        action = th.multinomial(probs, num_samples=1).item()
        log_prob = th.log(probs[0, action])
        return action, log_prob, probs

    def prepare_input(self, x):
        if not isinstance(x, th.Tensor):
            x = th.from_numpy(x.copy()).float().unsqueeze(0).to(self.device)
        x = x.permute(0, 3, 1, 2)
        x = th.relu(self.cnn1(x))
        x = th.relu(self.cnn2(x))
        x = self.pool(x)
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
def train_a2c(env_name, max_timesteps, gamma, lr, timesteps=3600):
    env = gym.make(env_name)
    env = PovOnlyObservation(env)
    env = ActionShaping(env)
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    model = ActorCritic(env.observation_space.shape[-1], 16, env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    optimizer = th.optim.RMSprop(model.parameters(), lr=lr)
    
    global_step = 0
    mean_rewards = []
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
            action, value, log_prob = model(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            optimizer.zero_grad()
            _, next_value, _ = model(obs)
            R = reward + gamma * next_value
            advantage = R - value
            policy_loss = -log_prob * advantage.detach()
            value_loss = advantage.pow(2)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            steps+=1
            global_step += 1
            local_rewards += reward
        global_reward += total_reward
        mean_rewards.append(local_rewards / steps)
        print(f"Episode {episode}, total reward: {total_reward}")
        episode += 1

    print("*********************************************")
    print(f"Training took {time.time() - time_start} seconds")
    print(f"Mean reward per episode: {global_reward / episode}")
    # save graph of mean rewards over episodes
    plt.plot(mean_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Mean Rewards")
    plt.title("Mean Rewards vs Episodes")
    plt.savefig("mean_rewards.png")
    env.close()
    # save model
    th.save(model.state_dict(), "model.pth")

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

if __name__ == "__main__":
    task = "MineRLTreechop-v0"
    train_a2c(task, 2000000, 0.99, 0.0001)


