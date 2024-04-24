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
from utils import PovOnlyObservation, ActionShaping, compute_gae, get_entropy_linear, make_env, dataset_action_batch_to_actions, initParser

DATA_DIR = "/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines"
BATCH_SIZE = 5

# Training loop for A2C
def train_a2c(
        max_timesteps, 
        gamma, 
        lr,
        env,
        experiment_name="a2c",
        load_model=False,
        entropy_start=0.0, # starting entropy value,
        annealing=False
    ):

    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    critic = Critic(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], device).to(device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    
    if load_model:
        actor.load_state_dict(th.load("bc_model.pth"))

    actor_optimizer = th.optim.RMSprop(actor.parameters(), lr=lr)
    critic_optimizer = th.optim.RMSprop(critic.parameters(), lr=lr)
    
    global_step = 0
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
        updates = 0

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

            # log rewards
            wandb.log({"reward": reward, "total_reward": total_reward, "steps": steps, "global_step": global_step})
            
            # update actor and critic every 10 steps
            if steps % 100 == 0 and total_reward > 0:
                with th.no_grad():
                    next_value = critic(state) if not done else 0
                returns = compute_gae(next_value, rewards, masks, values, gamma)        

                values = critic(th.stack(states).to(device).squeeze()).squeeze()
                log_probs = actor(th.stack(states).to(device).squeeze())[1]
                # clip log_probs to prevent NaN
                log_probs = th.clamp(log_probs, -1e8, 1e8)
                returns = th.tensor(returns, dtype=th.float32).to(device)
                advantages = returns - values
                # normalize advantages 
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # add entropy to loss
                if entropy_start > 0:
                    entropy_coef = get_entropy_linear(entropy_start, 0.0, global_step, max_timesteps) if annealing else entropy_start
                else:
                    entropy_coef = 0.0
                entropy = -th.mean(-log_probs)

                # define huber loss
                huber_loss = nn.SmoothL1Loss()

                # update critic
                critic_optimizer.zero_grad()
                critic_loss = advantages.pow(2).mean() + huber_loss(values, returns)
                critic_loss.backward()
                # clip gradients
                th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_optimizer.step()

                # update actor
                actor_optimizer.zero_grad()
                actor_loss = -(advantages.detach() * log_probs).mean() + entropy_coef * entropy
                actor_loss.backward()
                # clip gradients
                th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()
                # track updates
                updates += 1

                wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss, "steps": steps, "global_step": global_step, "updates": updates})
                
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
        local_rewards_arr.append(total_reward)

        wandb.log({"episode": episode, "total_reward": total_reward, "steps": steps, "global_step": global_step, "global_reward": global_reward})

        # checkpoint models
        if episode % 10 == 0:
            th.save(actor.state_dict(), f"{experiment_name}_actor_{episode}.pth")
            th.save(critic.state_dict(), f"{experiment_name}_critic_{episode}.pth")


    print("*********************************************")
    print(f"Training took {time.time() - time_start} seconds")
    print(f"Mean reward per episode: {global_reward / episode}")
    print(f"Total episodes: {episode}")
    print("Total rewards during training: ", global_reward)

    # Save graph of local rewards over episodes
    plt.plot(local_rewards_arr)
    plt.xlabel("Episodes")
    plt.ylabel("Local Rewards")
    plt.title("Local Rewards vs Episodes")
    plt.savefig("local_rewards.png")
    env.close()

    # Save models
    th.save(actor.state_dict(), f"{experiment_name}_actor.pth")
    th.save(critic.state_dict(), f"{experiment_name}_critic.pth")

# Test A2C model
def test_a2c(env, episodes, experiment_name, load_model=False):
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    if load_model:
        actor.load_state_dict(th.load(f"{experiment_name}_actor_10.pth"))
        
    actor.eval()

    global_reward = 0
    for episode in range(episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        # normalize observations and permute shape
        obs = obs.astype(np.float32) / 255.0
        state = th.tensor(obs.copy(), dtype=th.float32).to(device)
        state = state.unsqueeze(0).permute(0, 3, 1, 2)
        while not done:
            action, _, _ = actor(state)
            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32) / 255.0
            state = th.tensor(obs.copy(), dtype=th.float32).to(device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            total_reward += reward
        global_reward += total_reward
        print(f"Episode {episode}, total reward: {total_reward}")

    print("*********************************************")
    print(f"Mean reward per episode: {global_reward / episodes}")
    env.close()

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
    del data
    del network

# Test behavioral cloning model
def test_bc(env, episodes, model_path):
    # load model from model_path
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device, True).to(device)
    actor.load_state_dict(th.load(model_path))
    actor.eval()

    global_reward = 0

    for episode in range(episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            obs = obs.astype(np.float32) / 255.0
            state = th.tensor(obs.copy(), dtype=th.float32).to(device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            action, _, _ = actor(state)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        global_reward += total_reward
        print(f"Episode {episode}, total reward: {total_reward}")

    print("*********************************************")
    print(f"Mean reward: {global_reward / episodes}")
    print("Total rewards during testing: ", global_reward)

    env.close()

if __name__ == "__main__":
    parser = initParser()
    args = parser.parse_args()
    method = args.method
    if args.wandb == True:
        # # Initialize wandb - todo: next experiment will be w/out entropy for value loss
        wandb.init(project="minerl-a2c", config={
            "task": args.task,
            "max_timesteps": args.max_timesteps,
            "gamma": args.gamma,
            "learning_rate": args.learning_rate,
            "experiment_name": args.experiment_name,
            "load_model": args.load_model,
            "entropy_start": args.entropy_start,
        })

    if method == 'a2c':
        task = args.task
        max_timesteps = args.max_timesteps
        gamma = args.gamma
        lr = args.learning_rate
        experiment_name = args.experiment_name
        load_model = args.load_model
        entropy_start = args.entropy_start
        test = args.test
        annealing = args.annealing
        print(f"task: {task}, test: {test}, max_timesteps: {max_timesteps}, gamma: {gamma}, lr: {lr}, experiment_name: {experiment_name}, load_model: {load_model}, entropy_start: {entropy_start}")

        env = make_env(task, always_attack=True)
        if not test:
            train_a2c(max_timesteps, gamma, lr, env, experiment_name, load_model, entropy_start, annealing)
        else:
            test_a2c(env, 10, load_model)

    elif method == 'bc':
        task = args.task
        lr = args.learning_rate
        epochs = args.epochs
        env = make_env(task, always_attack=True)
        test = args.test
        if not test:
            train_bc(DATA_DIR, task, epochs, lr)
        else:
            test_bc(env, 10, "a2c_bc_model.pth")

    # env = make_env("MineRLTreechop-v0", always_attack=True)
    # train_bc(DATA_DIR, task, 5, 0.0001)
    # train_a2c(10000, .99, 0.0001, env, 'a2c_bc', 3600, True, 0.5)
    # test_bc(env, 10, "a2c_bc_model.pth")


