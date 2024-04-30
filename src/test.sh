#!/bin/bash

# Training A2C model solo
# Trains the A2C model on the MineRLTreechop-v0 environment 
# python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.99 --learning_rate 0.0001 --experiment_name a2c_training --method a2c --test false --load_model false --entropy_start 0.5

# Training A2C with Behavioral Cloning (A2C_BC)
# python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.999999 --learning_rate 0.0001 --experiment_name a2c_bc_training_selective_updates_noannealling --method a2c --load_model True --entropy_start 0.75 --wandb False

# Testing A2C with pretrained model (our best performing one)
python engine.py --task MineRLTreechop-v0 --method a2c --test true --load_model true --model_path a2c_bc.pth --episodes 100

# testing BC with pretrained model
# python engine.py --task MineRLTreechop-v0 --method bc --test true --load_model true --model_path bc_model.pth --episodes 100

# Training A2C with IQL
# python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.999999 --learning_rate 0.0001 --experiment_name a2c_bc_training_selective_updates_noannealling --method iq --load_model False --entropy_start 0.75 --wandb False

# Testing A2C+IQ with pretrained model
# python engine.py --task MineRLTreechop-v0 --method iq --test true --load_model true --model_path a2c_iq.pth --episodes 100
