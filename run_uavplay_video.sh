#!/bin/bash

# MBPO Play Script for UAV with Video Recording
# Usage: ./run_uavplay_video.sh <checkpoint_path> [video_length]

if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [video_length]"
    echo "Example: $0 runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt 500"
    exit 1
fi

CHECKPOINT_PATH=$1
VIDEO_LENGTH=${2:-200}

echo "Playing MBPO agent with checkpoint: $CHECKPOINT_PATH"
echo "Video length: $VIDEO_LENGTH steps"

./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint="$CHECKPOINT_PATH" \
    --num_envs=4 \
    --device=cuda:3 \
    --video \
    --video_length=$VIDEO_LENGTH
