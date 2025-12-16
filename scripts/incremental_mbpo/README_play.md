# MBPO Play Scripts

This directory contains scripts to play/evaluate trained MBPO agents.

## Files

- `play.py` - Main play script for MBPO agents
- `run_uavplay.sh` - Simple play script for UAV task
- `run_uavplay_video.sh` - Play script with video recording for UAV task

## Usage

### Basic Play (Headless)

```bash
# Using the simple script
./run_uavplay.sh runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt

# Or directly with python
./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint=runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt \
    --num_envs=16 \
    --device=cuda:3 \
    --headless
```

### Play with Video Recording

```bash
# Using the video script
./run_uavplay_video.sh runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt 500

# Or directly with python
./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint=runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt \
    --num_envs=4 \
    --device=cuda:3 \
    --video \
    --video_length=500
```

### Play with GUI (Non-headless)

```bash
./isaaclab.sh -p scripts/incremental_mbpo/play.py \
    --task=Isaac-Exploration-Rough-Drone-v0 \
    --checkpoint=runs/mbpo_ckpts/drone_20241201_120000/ckpt_step_10000.pt \
    --num_envs=4 \
    --device=cuda:3
```

## Arguments

- `--task`: Task name (e.g., Isaac-Exploration-Rough-Drone-v0)
- `--checkpoint`: Path to the checkpoint file (.pt)
- `--num_envs`: Number of parallel environments (default: 16)
- `--device`: Device to use (default: cuda:0)
- `--headless`: Run without GUI (default: False)
- `--video`: Record videos (default: False)
- `--video_length`: Length of recorded video in steps (default: 200)
- `--real-time`: Run in real-time mode (default: False)

## Output

The play script will:
1. Load the checkpoint and agent weights
2. Run the agent in the environment
3. Display real-time statistics (episodes, rewards, episode lengths)
4. Show final statistics when stopped (Ctrl+C)

## Statistics

- **Total episodes**: Number of completed episodes
- **Average reward**: Mean reward across all episodes
- **Average episode length**: Mean episode length
- **Best/Worst reward**: Best and worst episode rewards

## Notes

- Use fewer environments (4-8) for video recording to avoid performance issues
- Use more environments (16-32) for faster evaluation
- The script automatically handles episode completion and statistics
- Press Ctrl+C to stop evaluation and see final statistics
