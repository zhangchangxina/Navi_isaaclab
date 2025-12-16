export CUDA_VISIBLE_DEVICES=1
export WANDB_BASE_URL=https://api.bandw.top
# export WANDB_MODE=disabled

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/train.py \
  --task=Isaac-Exploration-Rough-Turtlebot-v0 \
  --num_envs=256 --max_iterations=10000 \
  --headless \
  --dyn_w_dyn=1.0 --dyn_w_r=1.0 --dyn_w_d=0.1 \
  --num_networks=1  --dynamics_lr=1e-3 --mb_buffer_size=1000000 --mb_batch_size=2048 --mb_update_every=24 --mb_update_steps=100  \
  --mb_virtual_envs=64 --mb_virt_steps_per_iter=0 --mb_init_from_buffer --mb_warmup_iters=10 \
  --use_stability_reward



# 不用模型数据：把 mb_virtual_envs 设为 0
#   --use_stability_reward \
