#!/bin/bash

# Test script to verify MBPO play functionality
# This script will look for the latest checkpoint and test the play script

echo "Testing MBPO play functionality..."

# Find the latest checkpoint
LATEST_CHECKPOINT=$(find runs/mbpo_ckpts -name "*.pt" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "No checkpoint found in runs/mbpo_ckpts/"
    echo "Please train a model first or specify a checkpoint path manually"
    exit 1
fi

echo "Found latest checkpoint: $LATEST_CHECKPOINT"

# Test basic play functionality
echo "Testing basic play (headless, 4 environments, 100 steps)..."
timeout 30s ./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint="$LATEST_CHECKPOINT" \
    --num_envs=4 \
    --device=cuda:3 \
    --headless

if [ $? -eq 0 ]; then
    echo "✅ Basic play test passed!"
else
    echo "❌ Basic play test failed!"
    exit 1
fi

echo "✅ All tests passed! MBPO play functionality is working correctly."
echo ""
echo "You can now use:"
echo "  ./run_uavplay.sh <checkpoint_path>"
echo "  ./run_uavplay_video.sh <checkpoint_path> [video_length]"
