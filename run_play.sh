#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Turtlebot-Play-v0 --num_envs=16 --checkpoint=logs/rsl_rl_incremental_model_based_ppo/drone_rough/2026-01-05_23-17-37_drone_experiment_mbppo/model_10000.pt