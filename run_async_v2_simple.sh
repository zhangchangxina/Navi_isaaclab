#!/bin/bash

# Simple script to run AsyncTrainer v2 without tabs command

echo "Starting Asynchronous MBPO Training v2 (RSL-RL style)..."

# Set environment variables
export ISAACLAB_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PYTHONPATH="${ISAACLAB_PATH}/source:${PYTHONPATH}"

# Run the training script
python scripts/incremental_mbpo/train_isaac_async_v2.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=4 \
    --num_steps_per_env=12 \
    --max_queue_size=3 \
    --device=cuda:3 \
    --headless \
    --log_interval=20 \
    --eval_interval=100 \
    --ckpt_interval=200
