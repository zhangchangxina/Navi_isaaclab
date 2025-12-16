#!/bin/bash

# Asynchronous MBPO Training Script with CUDA Debugging
# This script enables CUDA debugging to help identify device-side assertion errors

echo "Starting Asynchronous MBPO Training with CUDA Debugging..."

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run with debugging enabled
./isaaclab.sh -p scripts/incremental_mbpo/train_isaac_async.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=16 \
    --num_collectors=2 \
    --max_queue_size=500 \
    --collection_batch_size=64 \
    --device=cuda:3 \
    --headless \
    --log_interval=50 \
    --eval_interval=500 \
    --ckpt_interval=1000
