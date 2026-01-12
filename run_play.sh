#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=16 --checkpoint=logs/rsl_rl/drone_rough/2026-01-11_18-13-45_drone_experiment_ppo/model_9999.pt