#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=16 --checkpoint=logs/rsl_rl_incremental_model_based_ppo/drone_rough/2026-01-10_21-17-09_drone_experiment_mbppo/model_1000.pt