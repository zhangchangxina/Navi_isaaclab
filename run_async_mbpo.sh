#!/bin/bash

# Asynchronous MBPO Training Script
# This script demonstrates the separation of environment interaction from network training

echo "Starting Asynchronous MBPO Training..."

./isaaclab.sh -p scripts/incremental_mbpo/train_isaac_async.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=64 \
    --num_collectors=4 \
    --max_queue_size=1000 \
    --collection_batch_size=256 \
    --device=cuda:3 \
    --headless \
    --wandb \
    --log_interval=100 \
    --eval_interval=1000 \
    --ckpt_interval=2000
