#!/bin/bash
# 脚本路径：run_play_mb.sh

# 示例：回放 MB-PPO 策略 (开启增量动作)
# 请替换 --task 和 --checkpoint 为实际配置/路径

# 对于 UGV (Turtlebot)
# ./run_play_mb.sh --task Isaac-Exploration-Rough-Turtlebot-v0 --checkpoint <path_to_turtlebot_checkpoint.pt>

# 对于 UAV
# ./run_play_mb.sh --task Isaac-Exploration-Rough-Drone-v0 --checkpoint <path_to_drone_checkpoint.pt>
export CUDA_VISIBLE_DEVICES=3

python scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/play_mb.py \
    --task Isaac-Exploration-Rough-Drone-v0 \
    --num_envs 16 \
    --use_cbf --cbf_gamma 0.5 \
    --checkpoint logs/rsl_rl_incremental_model_based_ppo/drone_rough/2026-01-05_23-17-37_drone_experiment_mbppo/model_10000.pt \
    "$@"



    # --use_cbf \
    # --cbf_gamma 0.5 \
