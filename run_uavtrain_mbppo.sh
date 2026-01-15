# Pick a GPU via command line argument (default: 0)
# Usage: ./run_uavtrain_mbppo.sh [GPU_ID]
GPU_ID=${1:-0}
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export WANDB_BASE_URL=https://api.bandw.top
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HYDRA_FULL_ERROR=1
# export WANDB_MODE=disabled

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl_incremental_model_based_ppo/train.py \
  --task=Isaac-Exploration-Rough-Drone-v0 \
  --num_envs=256 --max_iterations=10000 \
  --headless \
  --dyn_w_dyn=1.0 --dyn_w_r=1.0 --dyn_w_d=0.1 \
  --num_networks=1  --dynamics_lr=1e-3 --mb_buffer_size=1000000 --mb_batch_size=2048 --mb_update_every=24 --mb_update_steps=100  \
  --mb_virtual_envs=64 --mb_virt_steps_per_iter=0 --mb_init_from_buffer --mb_warmup_iters=10 \
  --use_cbf --cbf_gamma 0.5 \
  --use_bc --bc_coef 0.1 --bc_decay 0.999 --bc_min 0.01 --bc_loss_type mse

# 不用模型数据：把 mb_virtual_envs 设为 0
# CBF 参数说明：
#   --use_cbf                    启用 CBF 安全滤波器
# BC 正则化参数说明 (Safe RL / ORL-RC 风格)：
#   --use_bc                     启用 BC 正则化 (需要同时启用 --use_cbf)
#   --bc_coef                    BC 正则化系数 (lambda)
#   --bc_decay                   每次 update 后的衰减因子 (设 <1 会逐渐减小)
#   --bc_min                     衰减后的最小 bc_coef 值
#   --bc_loss_type               损失类型: "mse" 或 "kl"
#                                - mse: ||u1 - u2||^2 (简单快速)
#                                - kl:  KL[π(·|s) || π_b(·|s)] (理论更优，Eq 9 风格)
