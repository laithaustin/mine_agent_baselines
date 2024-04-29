import minerl
import numpy as np
import torch as th
import torch.nn as nn
from tqdm import tqdm
from models.Actor import Actor
from utils import dataset_action_batch_to_actions
import matplotlib.pyplot as plt

# Training loop for behavioral cloning
def train_bc(
    dir, 
    task, 
    epochs, 
    learning_rate):
    """
    Use behavioral cloning to learn a policy from demonstrations.
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

    th.save(network.state_dict(), "bc_model.pth")
    # plot loss
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Behavioral Cloning Loss")
    plt.savefig("bc_loss.png")
    del data
    del network

# Test behavioral cloning model
def test_bc(env, episodes, model_path, render=False):
    # load model from model_path
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device, True).to(device)
    actor.load_state_dict(th.load(model_path))
    actor.eval()

    global_reward = 0
    rewards = []
    for episode in range(episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        while not done and steps < 4000:
            obs = obs.astype(np.float32) / 255.0
            state = th.tensor(obs.copy(), dtype=th.float32).to(device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            action, _, _ = actor(state)
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        global_reward += total_reward
        print(f"Episode {episode}, total reward: {total_reward}")

    print("*********************************************")
    print(f"Mean reward: {global_reward / episodes}")
    print(f"Variability: {np.std(rewards)}")
    env.close()
