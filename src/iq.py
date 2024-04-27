"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch as th
import torch.nn.functional as F
import torch.nn as nn
from models.Actor import Actor
from models.Critic import Critic
import numpy as np
import time
import minerl
from utils import dataset_action_batch_to_actions, compute_gae, get_entropy_linear, make_env
import matplotlib.pyplot as plt
import wandb
import gym
from multiprocessing import freeze_support
import os

def grad_pen(expert_states, expert_actions, policy_states, policy_actions, lambda_gp, critic):
    """
    Gradient penalty for Wasserstein_1 metric.
    """
    expert_states.requires_grad_(True)
    expert_actions.requires_grad_(True)
    policy_states.requires_grad_(True)
    policy_actions.requires_grad_(True)

    expert_Q = critic(expert_states, expert_actions)
    policy_Q = critic(policy_states, policy_actions)

    # Calculate gradients per data point
    expert_grad = th.autograd.grad(expert_Q, expert_states, grad_outputs=th.ones_like(expert_Q), create_graph=True, retain_graph=True)[0]
    policy_grad = th.autograd.grad(policy_Q, policy_states, grad_outputs=th.ones_like(policy_Q), create_graph=True, retain_graph=True)[0]

    # Compute per-example norms
    expert_grad_norm = expert_grad.view(expert_grad.size(0), -1).norm(2, dim=1)
    policy_grad_norm = policy_grad.view(policy_grad.size(0), -1).norm(2, dim=1)

    # Compute penalty
    grad_penalty = lambda_gp * ((expert_grad_norm - 1) ** 2).mean() + ((policy_grad_norm - 1) ** 2).mean()
    return grad_penalty

# Full IQ-Learn objective with other divergences and options
def iq_loss(agent,
            batch,
            div='hellinger',
            loss_type='value',
            regularize=False,
            lambda_gp=10.0,  
    ):
    """
    IQ Loss function given by authors of the inverse q-learning paper.
    """

    args = agent.args
    gamma = agent.gamma
    obs, action, env_reward, next_obs, done, is_expert = batch

    loss_dict = {}
    # calculate Q(s, a) and V(s')
    current_Q = agent.critic(obs, action)
    current_v = agent.critic(next_obs)
    next_v = agent.critic(next_obs)

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]

    with th.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if div == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif div == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = th.exp(-reward-1)
        elif div == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif div == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = th.exp(-reward)
        elif div == "js":
            # jensen–shannon
            phi_grad = th.exp(-reward)/(2 - th.exp(-reward))
        else:
            phi_grad = 1
    loss = -(phi_grad * reward).mean()
    loss_dict['softq_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if loss_type == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif loss_type == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    # add a gradient penalty to loss (Wasserstein_1 metric)
    gp_loss = grad_pen(obs[is_expert.squeeze(1), ...],
                                        action[is_expert.squeeze(1), ...],
                                        obs[~is_expert.squeeze(1), ...],
                                        action[~is_expert.squeeze(1), ...],
                                        lambda_gp, agent)
    loss_dict['gp_loss'] = gp_loss.item()
    loss += gp_loss

    if div == "chi" or args.method.chi:  # TODO: Deprecate method.chi argument for method.div
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    if regularize:
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict

# IQ-Learn critic update
def iq_critic_update(expert_batch, policy_batch, critic, critic_optimizer, batch_size=100, gamma=0.99, device="cuda"):
    # randomly choose a sample of size batch_size from expert data
    idx = np.random.choice(expert_batch.shape[0], 1, replace=False)
    # should get a sample of batch_size from expert data
    batch = batch[idx]
    # convert to tensors
    states = th.tensor(batch[:, 0]["pov"].squeeze().astype(np.float32), dtype=th.float32).to(device).permute(0, 3, 1, 2)
    actions = th.tensor(dataset_action_batch_to_actions(batch[:, 1]), dtype=th.float32).to(device)
    rewards = th.tensor(batch[:, 2], dtype=th.float32).to(device)
    next_states = th.tensor(batch[:, 3]["pov"].squeeze().astype(np.float32), dtype=th.float32).to(device).permute(0, 3, 1, 2)
    dones = th.tensor(batch[:, 4], dtype=th.float32).to(device)
    is_expert = th.tensor(np.ones((rewards.shape[0], 1)), dtype=th.float32).to(device)
    expert_batch = (states, actions, rewards, next_states, dones, is_expert)

    # get policy batch
    policy_states, policy_actions, policy_rewards, policy_next_states, policy_dones, policy_is_expert = policy_batch
    # combine policy and expert batches
    batch = (
        th.cat([states, policy_states], dim=0),
        th.cat([actions, policy_actions], dim=0),
        th.cat([rewards, policy_rewards], dim=0),
        th.cat([next_states, policy_next_states], dim=0),
        th.cat([dones, policy_dones], dim=0),
        th.cat([is_expert, policy_is_expert], dim=0)
    )

    # calculate IQ loss
    loss, loss_dict = iq_loss(critic, batch, div='kl_fix', loss='value', regularize=False)
    # update critic
    critic_optimizer.zero_grad()
    loss.backward()
    # clip gradients
    th.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    critic_optimizer.step()
    return loss.item()

# Training loop for A2C with inverse q learning - TODO: make generic
def train_a2c_iq(
        max_timesteps, 
        gamma, 
        lr,
        env,
        data_dir,
        batch_size=100,
        experiment_name="a2c",
        task = "MineRLTreechop-v0",
        load_model=False,
        entropy_start=0.0, # starting entropy value,
        annealing=False
    ):

    device = "cuda" if th.cuda.is_available() else "mps" if th.backends.mps.is_available() else "cpu"
    print("Using device: ", device)
    critic = Critic(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], device, num_actions=env.action_space.n).to(device)
    actor = Actor(env.observation_space.shape[-1], env.observation_space.shape[0], env.observation_space.shape[1], env.action_space.n, device).to(device)
    # load expert data from minerl dataset
    # get cpu count from os
    data = minerl.data.make(task,  data_dir=data_dir, num_workers=os.cpu_count()) # todo: update based on cpu cores
    iterator = minerl.data.BufferedBatchIter(data)
    expert_data = np.array([[s, a, r, ns, d] for s, a, r, ns, d in iterator.buffered_batch_iter(batch_size=100, num_epochs=1)])

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
        actions = []
        next_states = []
        dones = []
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
            actions.append(action)
            next_states.append(th.tensor(next_obs.copy(), dtype=th.float32).to(device).unsqueeze(0).permute(0, 3, 1, 2))
            dones.append(done)

            obs = next_obs
            total_reward += reward
            global_step += 1
            steps += 1

            # log rewards
            # wandb.log({"reward": reward, "total_reward": total_reward, "steps": steps, "global_step": global_step})
            
            # update actor and critic every batch_size steps
            if steps % batch_size == 0 and total_reward > 0:
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

                # update critic - updated to use IQ loss instead
                is_expert = th.tensor(np.zeros((len(rewards), 1)), dtype=th.float32).to(device)
                policy_batch = (th.stack(states).to(device).squeeze(), th.stack(actions).to(device).squeeze(), th.tensor(rewards, dtype=th.float32).to(device), th.stack(next_states).to(device).squeeze(), th.tensor(dones, dtype=th.float32).to(device), is_expert)
                critic_loss = iq_critic_update(expert_data, policy_batch, critic, critic_optimizer, batch_size=batch_size, gamma=gamma, device=device)

                # update actor
                actor_optimizer.zero_grad()
                actor_loss = -(advantages.detach() * log_probs).mean() + entropy_coef * entropy
                actor_loss.backward()
                # clip gradients
                th.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                actor_optimizer.step()
                # track updates
                updates += 1

                # wandb.log({"actor_loss": actor_loss, "critic_loss": critic_loss, "steps": steps, "global_step": global_step, "updates": updates})
                print(f"Actor loss: {actor_loss}, Critic loss: {critic_loss}, Steps: {steps}, Global step: {global_step}, Updates: {updates}")
                
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

        # wandb.log({"episode": episode, "total_reward": total_reward, "steps": steps, "global_step": global_step, "global_reward": global_reward})
            
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

# Train A2C with IQ-Learn
if __name__ == "__main__":
    freeze_support()
    env = make_env("MineRLTreechop-v0", always_attack=True)
    train_a2c_iq(
        max_timesteps=100000, 
        gamma=0.99, 
        lr=0.0001, 
        env=env, 
        data_dir="/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines/src", 
        batch_size=100, 
        experiment_name="a2c_iq", 
        task="MineRLTreechop-v0", 
        load_model=False,
        entropy_start=0.0,
        annealing=False
    )
    # # let's first test sampling the expert data
    # data = minerl.data.make("MineRLTreechop-v0",  data_dir="/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines/src", num_workers=4)
    # iterator = minerl.data.BufferedBatchIter(data)
    # batch = np.array([[s, a, r, ns, d] for s, a, r, ns, d in iterator.buffered_batch_iter(batch_size=100, num_epochs=1)])
    # # randomly choose batch_size samples from batch
    # # idx = np.random.choice(batch.shape[0], 100, replace=False)
    # # batch = batch[idx]
    # print("batch size: ", len(batch))
    # # grab first state
    # state = th.tensor(batch[0, 0]["pov"].squeeze().astype(np.float32), dtype=th.float32).unsqueeze(0).permute(0, 3, 1, 2)
    # print("state shape: ", state.shape)
    # # grab actions
    # print(dataset_action_batch_to_actions(batch[:, 1]).shape)
    # # actions = th.tensor(dataset_action_batch_to_actions(batch[0, 1]), dtype=th.float32)
    # # print("actions shape: ", actions.shape)