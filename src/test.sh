#!/bin/bash

# Training A2C
# Trains the A2C model on the MineRLTreechop-v0 environment 
# python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.99 --learning_rate 0.0001 --experiment_name a2c_training --method a2c --test false --load_model false --entropy_start 0.5

# Testing A2C
# Tests the A2C model on the MineRLTreechop-v0 environment using the previously trained model
# python engine.py --task MineRLTreechop-v0 --method a2c --test true --load_model true

# Training A2C with Behavioral Cloning (A2C_BC)
# Trains the A2C model integrated with Behavioral Cloning on the MineRLTreechop-v0 environment
python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.99 --learning_rate 0.0001 --experiment_name a2c_bc_training --method a2c --load_model True --entropy_start 0.5 --wandb False 

# Training both A2C and Behavioral Cloning
# Optionally train both models separatele
# python engine.py --task MineRLTreechop-v0 --epochs 5 --learning_rate 0.0001 --experiment_name a2c_and_bc_training --method bc
# python engine.py --task MineRLTreechop-v0 --max_timesteps 2000000 --gamma 0.99 --learning_rate 0.0001 --experiment_name a2c_and_bc_training --method a2c --test false --load_model true --entropy_start 0.5

