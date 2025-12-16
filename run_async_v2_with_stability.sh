#!/bin/bash

# Example script to run Async MBPO v2 with stability reward and cumulative action features
# This demonstrates the migrated functionality from sync MBPO

echo "Starting Async MBPO v2 with stability reward and cumulative action features..."

# Activate the virtual environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navi_isaaclab

# Run with stability reward shaping
python scripts/incremental_mbpo/train_isaac_async_v2.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --num_envs=8 \
    --num_steps_per_env=24 \
    --max_queue_size=10 \
    --device=cuda:0 \
    --use_stability_reward \
    --Q_scale=1.0 \
    --stability_coef=1e-3 \
    --use_cumulative_action \
    --use_lqr_baseline \
    --R_scale=0.1 \
    --wandb \
    --wandb_project="UAV_Navigation_Stability" \
    --wandb_run_name="AsyncMBPOv2_Stability_$(date +%Y%m%d_%H%M%S)" \
    --log_interval=1000 \
    --ckpt_interval=10000 \
    --eval_interval=5000

echo "Training completed!"
