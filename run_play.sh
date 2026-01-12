#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=16 --checkpoint=logs/rsl_rl/drone_rough/2026-01-12_12-12-12_drone_experiment_ppo/model_5000.pt