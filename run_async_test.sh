#!/bin/bash

# Simple Async Test Script
# This script tests the basic async components to identify CUDA issues

echo "Starting Simple Async Test..."

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Run simple test
./isaaclab.sh -p scripts/incremental_mbpo/test_async_simple.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=4 \
    --device=cuda:3 \
    --test_steps=50 \
    --headless
