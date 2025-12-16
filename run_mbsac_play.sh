#!/bin/bash

export CUDA_VISIBLE_DEVICES=3

./isaaclab.sh -p scripts/incremental_mbpo/play.py \
  --task=Isaac-Exploration-Rough-Drone-v0 \
  --video \
  --max_steps=1800 \
  --video_length=3000 \
  --num_envs=64 \
  --checkpoint=/media/nudt3090/XYQ/ZCX/navigation/Navi_IsaacLab/logs/mbpo/Isaac-Exploration-Rough-Drone-v0/MBPO_20251020_210031/ckpt_iter_3000.pt
