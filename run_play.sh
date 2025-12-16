#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Turtlebot-v0 --num_envs=64 --checkpoint=logs/rsl_rl_incremental_model_based_ppo/turtlebot_rough/2025-11-18_15-34-53_turtlebot_experiment_mbppo/model_9000.pt