#!/bin/bash

# MBPO Play Script for UAV
# Usage: ./run_uavplay.sh <checkpoint_path>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path>"
    echo "Example: $0 runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt"
    exit 1
fi

CHECKPOINT_PATH=$1

echo "Playing MBPO agent with checkpoint: $CHECKPOINT_PATH"

./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint="$CHECKPOINT_PATH" \
    --num_envs=16 \
    --device=cuda:3 \
    --headless
