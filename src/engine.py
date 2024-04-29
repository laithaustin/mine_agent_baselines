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
from bc import train_bc, test_bc
from a2c import train_a2c, test_a2c

DATA_DIR = "/Users/laithaustin/Documents/classes/rl/mine_agent/MineRL2021-Intro-baselines/src"
BATCH_SIZE = 5

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
        model_path = args.model_path
        episodes = args.episodes
        render = args.render
        print(f"task: {task}, test: {test}, max_timesteps: {max_timesteps}, gamma: {gamma}, lr: {lr}, experiment_name: {experiment_name}, load_model: {load_model}, entropy_start: {entropy_start}")

        env = make_env(task, always_attack=True)
        if not test:
            train_a2c(max_timesteps, gamma, lr, env, experiment_name, load_model, entropy_start, annealing)
        else:
            test_a2c(env, episodes, model_path, load_model, render=render)

    elif method == 'bc':
        task = args.task
        lr = args.learning_rate
        epochs = args.epochs
        model_path = args.model_path
        env = make_env(task, always_attack=True)
        test = args.test
        render = args.render
        episodes = args.episodes
        if not test:
            train_bc(DATA_DIR, task, epochs, lr)
        else:
            test_bc(env, episodes, model_path= model_path, render=render)

    # env = make_env("MineRLTreechop-v0", always_attack=True)
    # train_bc(DATA_DIR, task, 5, 0.0001)
    # train_a2c(10000, .99, 0.0001, env, 'a2c_bc', 3600, True, 0.5)
    # test_bc(env, 10, "a2c_bc_model.pth")


