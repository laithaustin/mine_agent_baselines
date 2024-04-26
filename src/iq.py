"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch as th
import torch.nn.functional as F
from models.Actor import Actor
from models.Critic import Critic
import numpy as np
import time
import minerl
from utils import dataset_action_batch_to_actions
import matplotlib.pyplot as plt
import wandb


# Full IQ-Learn objective with other divergences and options
import torch

def iq_loss(agent, 
            current_Q, 
            current_v, 
            next_v, 
            batch,
            div='hellinger',
            loss_strategy='value',
            regularize=False,
            gamma=0.9999,
            lambda_gp=10,
            alpha=0.1
    ):
    obs, next_obs, action, env_reward, done, is_expert = batch
    loss_dict = {}
    
    # Keep track of value of initial states
    v0 = agent.getV(obs[is_expert.squeeze(1), ...]).mean()
    loss_dict['v0'] = v0.item()

    # Calculate 1st term for IQ loss
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]

    with torch.no_grad():
        # Use different divergence functions
        if div == "hellinger":
            phi_grad = 1 / (1 + reward)**2
        elif div == "kl":
            phi_grad = torch.exp(-reward - 1)
        elif div == "js":
            phi_grad = torch.exp(-reward) / (2 - torch.exp(-reward))
        else:
            phi_grad = 1
    softq_loss = -(phi_grad * reward).mean()
    loss_dict['softq_loss'] = softq_loss.item()
    total_loss = softq_loss

    # Calculate 2nd term for IQ loss based on the chosen strategy
    if loss_strategy == "value_expert":
        value_loss = (current_v - y)[is_expert].mean()
        total_loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif loss_strategy == "value":
        value_loss = (current_v - y).mean()
        total_loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif loss_strategy == "v0":
        v0_loss = (1 - gamma) * v0
        total_loss += v0_loss
        loss_dict['v0_loss'] = v0_loss.item()

    # Add a gradient penalty to loss (Wasserstein_1 metric)
    if regularize:
        gp_loss = agent.grad_pen(obs[is_expert.squeeze(1), ...],
                                 action[is_expert.squeeze(1), ...],
                                 obs[~is_expert.squeeze(1), ...],
                                 action[~is_expert.squeeze(1), ...],
                                 lambda_gp)
        loss_dict['gp_loss'] = gp_loss.item()
        total_loss += gp_loss

    if div == "chi":
        # χ2 divergence, calculate the regularization term using expert states
        chi2_loss = 1 / (4 * alpha) * (reward**2)[is_expert].mean()
        total_loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict

# Training loop for A2C with inverse q learning - TODO: make generic
def train_a2c_iq(
        max_timesteps, 
        gamma, 
        lr,
        env,
        data_dir,
        experiment_name="a2c",
        task = "MineRLTreechop-v0",
        load_model=False,
        entropy_start=0.0, # starting entropy value,
        annealing=False
    ):

    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    critic = Critic(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], device).to(device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    data = minerl.data.make(task,  data_dir=data_dir, num_workers=4)

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
        total_reward = 0

        # use expert data every other episode
        if episode % 2 == 0:
            total_reward, steps = train_expert()           
        else:
            total_reward, steps = train_loop()

        episode += 1
        global_step += steps
        print(f"Episode {episode}, total reward: {total_reward}")
        global_reward += total_reward
        local_rewards_arr.append(total_reward)

        wandb.log({"episode": episode, "total_reward": total_reward, "steps": steps, "global_step": global_step, "global_reward": global_reward})

        # checkpoint models
        if episode % 10 == 0:
            th.save(actor.state_dict(), f"{experiment_name}_actor_iq_{episode}.pth")
            th.save(critic.state_dict(), f"{experiment_name}_critic_iq_{episode}.pth")


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
    th.save(actor.state_dict(), f"{experiment_name}_actor_iq.pth")
    th.save(critic.state_dict(), f"{experiment_name}_critic_iq.pth") 

# Training loop using interactions with environment for IQ-Learn
def train_loop():
    # TODO: Implement training loop for IQ-Learn
    pass

# Training loop using expert data for IQ-Learn
def train_expert(
        data, # minerl expert data
        device,
        model,
        stream_name,
        optimizer,
    ):
    """
    One batch of expert data collection.
    """
    # sample one trajectory from the expert data
    state, action, reward, next_state, done = data.load_data(stream_name)
    state = state["pov"].squeeze().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    action = dataset_action_batch_to_actions(action)
    next_state = next_state["pov"].squeeze().astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    done = done.squeeze()
    reward = reward.squeeze()

    # convert to tensors
    state = th.tensor(state, dtype=th.float32).to(device)
    action = th.tensor(action, dtype=th.long).to(device)
    next_state = th.tensor(next_state, dtype=th.float32).to(device)
    done = th.tensor(done, dtype=th.float32).to(device)
    reward = th.tensor(reward, dtype=th.float32).to(device)

    batch = (state, next_state, action, reward, done, True)

    # get critic values
    current_v = model(state)
    next_v = model(next_state)
    current_q = model(state, action)

    # compute loss
    loss, loss_dict = iq_loss(model, current_q, current_v, next_v, batch)

    # update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()