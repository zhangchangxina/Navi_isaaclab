# export CUDA_VISIBLE_DEVICES=3
export WANDB_BASE_URL=https://api.bandw.top
# export WANDB_MODE=disabled



# ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Exploration-Rough-Drone-v0 --num_envs=256 --max_iterations=5000

./isaaclab.sh -p scripts/incremental_mbpo/train_isaac_mbpo.py --task=Isaac-Exploration-Rough-Drone-v0 --headless --wandb --device=cuda:3
