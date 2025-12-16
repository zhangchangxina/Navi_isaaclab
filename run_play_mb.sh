#!/bin/bash
# 脚本路径：run_play_mb.sh

# 示例：回放 MB-PPO 训练的 UGV 策略 (开启增量动作)
# 请替换 --load_run 和 --checkpoint 为实际路径

# 对于 UGV
# ./run_play_mb.sh --task Isaac-Velocity-Rough-Turtlebot-v0 --load_run 2025-xx-xx_xx-xx-xx --checkpoint model_xxx.pt

# 对于 UAV
# ./run_play_mb.sh --task Isaac-Velocity-Rough-Drone-v0 ...

python scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/play_mb.py \
    --task Isaac-Velocity-Rough-Turtlebot-v0 \
    --num_envs 4 \
    --use_incremental_actions \
    "$@"



