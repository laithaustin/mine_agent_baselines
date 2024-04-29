import time
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
from models.Actor import Actor
from models.Critic import Critic
from utils import compute_gae, get_entropy_linear
import minerl

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
            states.append(state)

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
def test_a2c(env, episodes, model_name, load_model=False, render=False):
    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    if load_model:
        actor.load_state_dict(th.load(f"{model_name}.pth"))
        
    actor.eval()

    global_reward = 0
    rewards = []
    for episode in range(episodes):
        done = False
        total_reward = 0
        obs = env.reset()
        steps = 0
        # normalize observations and permute shape
        obs = obs.astype(np.float32) / 255.0
        state = th.tensor(obs.copy(), dtype=th.float32).to(device)
        state = state.unsqueeze(0).permute(0, 3, 1, 2)
        while not done and steps < 4000:
            action, _, _ = actor(state)
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            obs = obs.astype(np.float32) / 255.0
            state = th.tensor(obs.copy(), dtype=th.float32).to(device)
            state = state.unsqueeze(0).permute(0, 3, 1, 2)
            total_reward += reward
            steps += 1
        rewards.append(total_reward)
        global_reward += total_reward
        print(f"Episode {episode}, total reward: {total_reward}")

    print("*********************************************")
    print(f"Mean reward per episode: {global_reward / episodes}")
    print(f"Variability: {np.std(rewards)}")
    env.close()