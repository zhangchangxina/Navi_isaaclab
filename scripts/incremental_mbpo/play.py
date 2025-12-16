# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an MBPO agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import time
import torch
import numpy as np

# Add the repository root to the path
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "source")
sys.path.insert(0, _SRC_DIR)

try:
    from omni.isaac.lab.app import AppLauncher  # Isaac Lab canonical path
except Exception:
    from isaaclab.app import AppLauncher  # fallback to local package name

# Ensure repository root is on sys.path for local package imports (incremental_mbpo)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

def parse_args():
    parser = argparse.ArgumentParser(description="Play with MBPO agent")
    parser.add_argument("--task", type=str, required=True, help="Task name")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of environments")
    parser.add_argument("--video", action="store_true", default=False, help="Record videos")
    parser.add_argument("--video_length", type=int, default=200, help="Length of recorded video (in steps)")
    parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    
    # Isaac app
    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    if getattr(args_cli, "render", False):
        args_cli.enable_cameras = True
    sys.argv = [sys.argv[0]] + hydra_args
    return args_cli

def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac task registry and helpers AFTER launching the app
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from scripts.incremental_mbpo.isaac_adapter import IsaacVecEnvAdapter
    from scripts.incremental_mbpo.algo.sac import SAC

    print(f"[INFO] Playing MBPO agent for task: {args.task}")
    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] Device: {args.device}")
    print(f"[INFO] Number of environments: {args.num_envs}")

    # Load environment configuration
    env_cfg = load_cfg_from_registry(args.task.split(":")[-1], "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device
    env_cfg.seed = 1234

    # Disable debug visualization
    try:
        if getattr(env_cfg.scene, "plane", None) is not None:
            env_cfg.scene.plane.debug_vis = False
        if hasattr(env_cfg, "viewer"):
            env_cfg.viewer.enable = not args.headless
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "debug_vis"):
            env_cfg.curriculum.debug_vis = False
    except Exception:
        pass

    # Create environment
    vec_env = IsaacVecEnvAdapter(args.task, env_cfg=env_cfg, device=args.device, render=not args.headless)
    obs_mat, _ = vec_env.reset(seed=1234)
    obs_dim = obs_mat.shape[-1]
    act_dim = vec_env.act_dim

    print(f"[INFO] Observation dimension: {obs_dim}")
    print(f"[INFO] Action dimension: {act_dim}")

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    print(f"[INFO] Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Extract agent configuration from checkpoint
    if "cfg" in checkpoint:
        cfg = checkpoint["cfg"]
        print(f"[INFO] Checkpoint task: {cfg.get('task', 'Unknown')}")
        print(f"[INFO] Checkpoint device: {cfg.get('device', 'Unknown')}")
        print(f"[INFO] Checkpoint num_envs: {cfg.get('num_envs', 'Unknown')}")

    # Create SAC agent
    class SACArgs:
        def __init__(self, config, device):
            self.gamma = config.get('gamma', 0.99)
            self.tau = config.get('tau', 0.005)
            self.alpha = config.get('alpha', 0.2)
            self.policy = "Gaussian"
            self.target_update_interval = config.get('target_update_interval', 1)
            self.automatic_entropy_tuning = config.get('automatic_entropy_tuning', True)
            self.cuda = device.startswith("cuda")
            self.lr = config.get('lr', 1e-3)
            self.critic_lr = config.get('critic_lr', 1e-3)
            self.policy_lr = config.get('policy_lr', 1e-3)
            self.alpha_lr = config.get('alpha_lr', 1e-3)
            self.policy_hidden_dims = config.get('policy_hidden_dims', [256, 256])
            self.critic_hidden_dims = config.get('critic_hidden_dims', [256, 256])
            self.num_epoch = config.get('num_epoch', 1)
            self.epoch_length = config.get('epoch_length', 1000)

    # Default configuration for SAC
    default_config = {
        'gamma': 0.99,
        'tau': 0.005,
        'alpha': 0.2,
        'lr': 1e-3,
        'critic_lr': 1e-3,
        'policy_lr': 1e-3,
        'alpha_lr': 1e-3,
        'policy_hidden_dims': [256, 256],
        'critic_hidden_dims': [256, 256],
        'target_update_interval': 1,
        'automatic_entropy_tuning': True,
        'num_epoch': 1,
        'epoch_length': 1000,
    }

    sac_args = SACArgs(default_config, args.device)

    class _ActSpace:
        def __init__(self, dim):
            self.shape = (dim,)
            self.high = np.ones(dim, dtype=np.float32)
            self.low = -np.ones(dim, dtype=np.float32)

    act_space = _ActSpace(act_dim)
    obs_space_shape = (obs_dim,)

    # Create SAC agent
    agent = SAC(obs_space_shape, act_space, sac_args)

    # Load agent weights from checkpoint
    if "agent" in checkpoint:
        agent.load_state_dict(checkpoint["agent"])
        print("[INFO] Loaded agent weights from checkpoint")
    else:
        print("[WARNING] No agent weights found in checkpoint")

    # Set agent to evaluation mode
    agent.eval()

    print("[INFO] Starting evaluation...")
    print("[INFO] Press Ctrl+C to stop")

    # Evaluation loop
    episode_rewards = []
    episode_lengths = []
    current_episode_rewards = np.zeros(args.num_envs)
    current_episode_lengths = np.zeros(args.num_envs)
    episode_count = 0

    try:
        step = 0
        while True:
            # Get actions from agent
            with torch.no_grad():
                actions = []
                for i in range(args.num_envs):
                    obs = obs_mat[i]
                    action = agent.select_action(obs)
                    actions.append(action)
                actions = np.array(actions)

            # Step environment
            obs_mat, rewards, dones, infos = vec_env.step(actions)

            # Update episode statistics
            current_episode_rewards += rewards
            current_episode_lengths += 1

            # Check for episode completion
            for i in range(args.num_envs):
                if dones[i]:
                    episode_rewards.append(current_episode_rewards[i])
                    episode_lengths.append(current_episode_lengths[i])
                    episode_count += 1
                    current_episode_rewards[i] = 0
                    current_episode_lengths[i] = 0

            step += 1

            # Print statistics every 100 steps
            if step % 100 == 0:
                if episode_count > 0:
                    avg_reward = np.mean(episode_rewards[-min(10, len(episode_rewards)):])
                    avg_length = np.mean(episode_lengths[-min(10, len(episode_lengths)):])
                    print(f"[Step {step}] Episodes: {episode_count}, Avg Reward (last 10): {avg_reward:.3f}, Avg Length (last 10): {avg_length:.1f}")

            # Real-time control
            if args.real_time:
                time.sleep(0.01)  # 100 Hz

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation stopped by user")

    # Final statistics
    if episode_count > 0:
        print(f"\n[INFO] Final Statistics:")
        print(f"  Total episodes: {episode_count}")
        print(f"  Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
        print(f"  Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
        print(f"  Best reward: {np.max(episode_rewards):.3f}")
        print(f"  Worst reward: {np.min(episode_rewards):.3f}")

    # Close environment
    vec_env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
