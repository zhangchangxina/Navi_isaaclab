export CUDA_VISIBLE_DEVICES=3
export WANDB_BASE_URL=https://api.bandw.top

./isaaclab.sh -p scripts/incremental_mbpo/train_isaac_async_v2.py \
       --task=Isaac-Exploration-Rough-Drone-v0 \
       --num_envs=8 \
       --num_steps_per_env=24 \
       --use_stability_reward \
       --use_cumulative_action \
       --wandb \
       --headless