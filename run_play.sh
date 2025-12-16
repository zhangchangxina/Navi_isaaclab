#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=128 --checkpoint=/media/nudt3090/XYQ/ZCX/navigation/Navi_IsaacLab/logs/rsl_rl/drone_rough/2025-08-06_21-09-03_drone_experiment/model_4999.pt --video --video_length=300