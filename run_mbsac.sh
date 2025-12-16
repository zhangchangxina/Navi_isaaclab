export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=https://api.bandw.top

./isaaclab.sh -p scripts/incremental_mbpo/train_mbpo.py \
       --max_iterations=10000 \
       --num_envs=1024 \
       --num_steps_per_env=8 \
       --batch_size=1024 \
       --updates_per_cycle=50 \
       --use_rsl_rl_updates \
       --log_interval=1 \
       --eval_interval=100 \
       --ckpt_interval=1000 \
       --task=Isaac-Exploration-Rough-Drone-v0 \
       --min_pool_size=100000 \
       --wandb \
       --headless \
