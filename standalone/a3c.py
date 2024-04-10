import warnings 
from typing import Any, ClassVar, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

import multiprocessing as mp

class A3C(OnPolicyAlgorithm):
    """
    Asynchronous Advantage Actor Critic (A3C)

    Papers: https://arxiv.org/pdf/1602.01783.pdf

    Summary:
    - A3C is an asychronous version of the A2C algorithm
    - Online algorithm that learns a policy and a value function
    - It uses multiple workers that interact with the environment in parallel
    - Each worker collects experience and updates the global network
    - The global network is updated with the gradients from the workers 

    # We're going to follow the code standard set by the stable_baselines library
    # specify params here eventually
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        worker_threads: int = -1,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        # using RMSprop as the optimizer
        self.policy.optimizer = th.optim.RMSprop(self.policy.parameters(), lr=learning_rate)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # TODO: maybe clip policy and value networks
        # # Initialize schedules for policy/value clipping
        # self.clip_range = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    class Worker(mp.Process):
        def __init__(self, worker_id, global_a3c, env, queue, T_max=1000):
            self.worker_id = worker_id
            self.global_a3c = global_a3c
            self.env = env
            self.policy = ActorCriticPolicy(env.observation_space, env.action_space, **global_a3c.policy_kwargs)
            self.T_max = T_max
            # initial thread step counter
            self.t = 0
            # queue to store gradients
            self.queue = queue

        """
        Since we are using multiprocessing, we will need to run a single episode in each worker, return the gradients
        and update the global network with the orchestrator. 
        """
        def run(self):
            # set device
            self.policy.to(self.global_a3c.device)
            # set training mode
            self.policy.set_training_mode(True)
            t_start = self.t
            done = False
            # get initial state
            state = self.env.reset()
            # reset gradients
            d_theta_actor = 0
            d_theta_critic = 0
            # synchronize weights with the global network
            self.policy.value_net.load_state_dict(self.global_a3c.policy.value_net.state_dict())
            self.policy.actor_net.load_state_dict(self.global_a3c.policy.actor_net.state_dict())
            
            # collect trajectory
            traj = []
            while not done and self.t - t_start < self.T_max:
                # perform action according to the policy
                action, log_prob, value = self.policy.forward(state)
                next_state, reward, done, _ = self.env.step(action)
                traj.append((state, action, reward, log_prob, value))
                self.t += 1
                self.global_a3c.T += 1
                state = next_state

            # bootstrap from last state
            R = 0 if done else traj[-1][-1]

            actor_losses, critic_losses = [], []
            # compute R and accumulate gradients
            for state, action, reward, log_prob, value in traj[::-1]:
                R = reward + self.global_a3c.gamma * R
                adv = R - value
                # compute losses
                actor_loss = -adv * log_prob
                critic_loss = adv ** 2

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                # accumulate gradients
                d_theta_actor += th.autograd.grad(actor_loss, self.policy.actor_net.parameters())
                d_theta_critic += th.autograd.grad(critic_loss, self.policy.value_net.parameters())
            
            # add the gradients to the queue
            self.queue.put(d_theta_actor, d_theta_critic, self.worker_id, self.t, np.mean(actor_losses), np.mean(critic_losses))

    def train(self) -> None:
        # Move policy to correct device
        self.policy.to(self.device)
        # train mode on
        self.policy.set_training_mode(True)
        # compute number of processors we can run this on
        n_workers = mp.cpu_count() if self.worker_threads == -1 else self.worker_threads
        T = 0
        T_max = self._total_timesteps

        # create workers and run processes
        workers = [self.Worker(i, self, self.env, self.queue) for i in range(n_workers)]

        while T < self._total_timesteps:
            for worker in workers:
                worker.start()

            for worker in workers:
                worker.join()
                d_theta_actor, d_theta_critic, worker_id, t, actor_loss, critic_loss = self.queue.get()
                # accumulate gradients using optimizer RMSprop
                self.optimizer.zero_grad()
                for param, grad in zip(self.policy.actor_net.parameters(), d_theta_actor):
                    param.grad = grad
                for param, grad in zip(self.policy.value_net.parameters(), d_theta_critic):
                    param.grad = grad
                self.optimizer.step()
                print(f"Worker {worker_id} step {t}: actor loss {actor_loss}, critic loss {critic_loss}")

    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4, eval_env: GymEnv = None, eval_freq: int = -1, n_eval_episodes: int = 5, tb_log_name: str = "A3C", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True) -> "A3C":
        # create multiple workers and organzie them
        return super().learn(total_timesteps, callback, log_interval, eval_env, eval_freq, n_eval_episodes, tb_log_name, eval_log_path, reset_num_timesteps)

