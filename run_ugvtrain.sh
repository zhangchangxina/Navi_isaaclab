export CUDA_VISIBLE_DEVICES=0
export WANDB_BASE_URL=https://api.bandw.top


./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Exploration-Rough-Turtlebot-v0 --num_envs=256 --max_iterations=10000 --headless
