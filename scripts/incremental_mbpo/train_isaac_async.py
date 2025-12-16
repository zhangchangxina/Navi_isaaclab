"""
Asynchronous MBPO training script that separates environment interaction from network training.
This maximizes the utilization of Isaac Lab's parallel environments.
"""

import argparse
import sys
import os
import numpy as np
import time
from typing import Optional
import json
import torch

# Ensure repo source is on sys.path so `isaaclab` (or omni.isaac.lab) is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "source")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from omni.isaac.lab.app import AppLauncher  # Isaac Lab canonical path
except Exception:
    from isaaclab.app import AppLauncher  # fallback to local package name

# Ensure repository root is on sys.path for local package imports (incremental_mbpo)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.incremental_mbpo.algo.replay_memory import ReplayMemory
from scripts.incremental_mbpo.algo.sac import SAC
from scripts.incremental_mbpo.async_trainer import create_async_trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Asynchronous MBPO training on Isaac task")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel Isaac sub-environments")
    
    # Async training specific
    parser.add_argument("--num_collectors", type=int, default=4, help="Number of data collection threads")
    parser.add_argument("--max_queue_size", type=int, default=1000, help="Maximum size of data queues")
    parser.add_argument("--collection_batch_size", type=int, default=256, help="Batch size for data collection")
    
    # checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="runs/async_mbpo_ckpts", help="Directory to save checkpoints")
    parser.add_argument("--ckpt_interval", type=int, default=None, help="Save checkpoint every N training steps")
    parser.add_argument("--eval_interval", type=int, default=None, help="Run evaluation every N training steps")
    parser.add_argument("--log_interval", type=int, default=None, help="Log training info every N training steps")
    
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default=None, help="W&B project name (auto by task if None)")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    
    # Learning rates (override config file defaults)
    parser.add_argument("--critic_lr", type=float, default=None, help="Critic learning rate")
    parser.add_argument("--policy_lr", type=float, default=None, help="Policy learning rate")
    parser.add_argument("--alpha_lr", type=float, default=None, help="Alpha learning rate")
    parser.add_argument("--dynamics_lr", type=float, default=None, help="Dynamics model learning rate")
    
    # UTD options (updates per data): factors per environment
    parser.add_argument("--utd_critic_per_env", type=int, default=4, help="Critic updates per environment per step (UTD)")
    parser.add_argument("--utd_actor_per_env", type=int, default=1, help="Actor updates per environment per step (UTD)")
    parser.add_argument("--utd_dynamic_per_env", type=int, default=4, help="Dynamics model updates per environment per step (UTD)")
    
    # Isaac app (AppLauncher will add device argument automatically)
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
    from scripts.incremental_mbpo.algo.incremental_dynamics import IncrementalDynamicsModel, incremental_dynamics_loss

    # Avoid spawning visualization USDs in headless mode
    try:
        from isaaclab.terrains.terrain_importer import TerrainImporter
        from isaaclab.markers.visualization_markers import VisualizationMarkers

        def _no_vis(self, debug_vis: bool):
            self.debug_vis = False
            return

        TerrainImporter.set_debug_vis = _no_vis  # type: ignore[attr-defined]

        def _noop_add_markers(self, markers_cfg):
            return

        VisualizationMarkers._add_markers_prototypes = _noop_add_markers  # type: ignore[attr-defined]
    except Exception:
        pass

    # Isaac env cfg (vectorized)
    env_cfg = load_cfg_from_registry(args.task.split(":")[-1], "env_cfg_entry_point")
    num_envs = getattr(args, "num_envs", None)
    if num_envs is None:
        # Try to get from task config
        task_rl_cfg = _load_task_rl_cfg(args.task)
        if task_rl_cfg is not None:
            num_envs = getattr(task_rl_cfg, "num_envs", None)
    env_cfg.scene.num_envs = int(num_envs) if num_envs else 8
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed

    try:
        if getattr(env_cfg.scene, "plane", None) is not None:
            env_cfg.scene.plane.debug_vis = False
        if getattr(env_cfg.scene, "terrain", None) is not None:
            env_cfg.scene.terrain.debug_vis = False
        if getattr(env_cfg.scene, "lidar_scanner", None) is not None:
            env_cfg.scene.lidar_scanner.debug_vis = False
        if hasattr(env_cfg, "viewer"):
            env_cfg.viewer.enable = False
        if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "debug_vis"):
            env_cfg.curriculum.debug_vis = False
    except Exception:
        pass

    # Build vectorized adapter
    vec_env = IsaacVecEnvAdapter(args.task, env_cfg=env_cfg, device=args.device, render=getattr(args, "render", False))
    obs_mat, _ = vec_env.reset(seed=args.seed)
    obs_dim = obs_mat.shape[-1]
    act_dim = vec_env.act_dim

    # Load task configuration
    def _load_task_rl_cfg(task_id: str):
        try:
            if "drone" in task_id.lower() or "uav" in task_id.lower():
                from isaaclab_tasks.manager_based.exploration.velocity.config.drone.agents.mbpo_cfg import (
                    DroneSACCfg as RLCfg,
                )
            else:
                from isaaclab_tasks.manager_based.exploration.velocity.config.turtlebot.agents.mbpo_cfg import (
                    TurtlebotSACCfg as RLCfg,
                )
            return RLCfg()
        except Exception:
            return None

    task_rl_cfg = _load_task_rl_cfg(args.task)

    def get_config(task_cfg, args, env_cfg=None):
        if task_cfg is None:
            raise RuntimeError(f"No configuration found for task")
        config = {
            'gamma': task_cfg.gamma,
            'tau': task_cfg.tau,
            'alpha': task_cfg.alpha,
            'lr': 1e-3,  # Default learning rate (fallback)
            'critic_lr': getattr(task_cfg, 'critic_lr', 1e-3),
            'policy_lr': getattr(task_cfg, 'policy_lr', 1e-3),
            'alpha_lr': getattr(task_cfg, 'alpha_lr', 1e-3),
            'dynamics_lr': getattr(task_cfg, 'dynamics_lr', 1e-3),
            'policy_hidden_dims': task_cfg.policy_hidden_dims,
            'critic_hidden_dims': task_cfg.critic_hidden_dims,
            'automatic_entropy_tuning': task_cfg.automatic_entropy_tuning,
            'target_update_interval': task_cfg.target_update_interval,
            'num_networks': task_cfg.num_networks,
            'num_elites': task_cfg.num_elites,
            'dynamics_hidden_dims': task_cfg.dynamics_hidden_dims,
            'rollout_batch_size': task_cfg.rollout_batch_size,
            'rollout_min_length': task_cfg.rollout_min_length,
            'rollout_max_length': task_cfg.rollout_max_length,
            'rollout_min_epoch': task_cfg.rollout_min_epoch,
            'rollout_max_epoch': task_cfg.rollout_max_epoch,
            'replay_size': task_cfg.replay_size,
            'min_pool_size': task_cfg.min_pool_size,
            'real_ratio': task_cfg.real_ratio,
            'policy_train_batch_size': task_cfg.policy_train_batch_size,
            'init_exploration_steps': getattr(task_cfg, 'init_exploration_steps', 10000),
            'epoch_length': getattr(task_cfg, 'epoch_length', 1000),
            'num_epoch': getattr(task_cfg, 'num_epoch', 1000),
            'max_path_length': 1000,  # Default fallback, will be overridden by env_cfg
            'dyn_w_dyn': getattr(task_cfg, 'dyn_w_dyn', 1.0),
            'dyn_w_r': getattr(task_cfg, 'dyn_w_r', 1.0),
            'dyn_w_d': getattr(task_cfg, 'dyn_w_d', 0.1),
            'dyn_done_threshold': getattr(task_cfg, 'dyn_done_threshold', 0.5),
        }
        if env_cfg is not None:
            episode_length_s = getattr(env_cfg, 'episode_length_s', 180.0)
            sim_dt = getattr(env_cfg.sim, 'dt', 0.01)
            decimation = getattr(env_cfg.sim, 'decimation', 10)
            episode_length_steps = int(episode_length_s / (sim_dt * decimation))
            if config['rollout_max_length'] < episode_length_steps * 0.1:
                config['rollout_max_length'] = max(50, int(episode_length_steps * 0.15))
            config['max_path_length'] = episode_length_steps
            config['epoch_length'] = episode_length_steps
            config['episode_length_steps'] = episode_length_steps
            config['episode_length_s'] = episode_length_s
            config['sim_dt'] = sim_dt
            config['decimation'] = decimation
        cfg_num_steps = getattr(task_cfg, 'num_steps', None)
        if cfg_num_steps is not None:
            config['num_steps'] = int(cfg_num_steps)
        else:
            config['num_steps'] = int(config['num_epoch'] * config['epoch_length'])
        if args is not None:
            if hasattr(args, 'critic_lr') and args.critic_lr is not None:
                config['critic_lr'] = args.critic_lr
            else:
                config['critic_lr'] = config['lr']
            if hasattr(args, 'policy_lr') and args.policy_lr is not None:
                config['policy_lr'] = args.policy_lr
            else:
                config['policy_lr'] = config['lr']
            if hasattr(args, 'alpha_lr') and args.alpha_lr is not None:
                config['alpha_lr'] = args.alpha_lr
            else:
                config['alpha_lr'] = config['lr']
            if hasattr(args, 'dynamics_lr') and args.dynamics_lr is not None:
                config['dynamics_lr'] = args.dynamics_lr
            else:
                config['dynamics_lr'] = config['lr']
            # Async training specific configs
            config['num_collectors'] = getattr(args, 'num_collectors', 4)
            config['max_queue_size'] = getattr(args, 'max_queue_size', 1000)
            config['collection_batch_size'] = getattr(args, 'collection_batch_size', 256)
            # logging and checkpointing intervals from CLI or config
            if hasattr(args, 'log_interval') and args.log_interval is not None:
                config['log_interval'] = int(args.log_interval)
            else:
                config['log_interval'] = getattr(task_cfg, 'log_interval', 1000)
            if hasattr(args, 'eval_interval') and args.eval_interval is not None:
                config['eval_interval'] = int(args.eval_interval)
            else:
                config['eval_interval'] = getattr(task_cfg, 'eval_interval', 10000)
            if hasattr(args, 'ckpt_interval') and args.ckpt_interval is not None:
                config['ckpt_interval'] = int(args.ckpt_interval)
            else:
                config['ckpt_interval'] = getattr(task_cfg, 'ckpt_interval', 50000)
        return config

    config = get_config(task_rl_cfg, args, env_cfg)
    print(f"Environment settings: episode_length_s={config.get('episode_length_s', 'N/A')}, dt={config.get('sim_dt', 'N/A')}, decimation={config.get('decimation', 'N/A')}")
    print(f"Calculated episode_length_steps: {config.get('episode_length_steps', 'N/A')}")
    print(f"Auto-adjusted rollout_max_length: {config['rollout_max_length']}")
    print(f"Async training config: num_collectors={config['num_collectors']}, max_queue_size={config['max_queue_size']}")

    # Optional: init wandb
    wandb_run: Optional[object] = None
    if getattr(args, "wandb", False):
        try:
            import wandb  # type: ignore
            auto_proj = None
            tl = args.task.lower()
            if "turtlebot" in tl or "ugv" in tl:
                auto_proj = "UGV_Navigation"
            elif "drone" in tl or "uav" in tl:
                auto_proj = "UAV_Navigation"
            project_name = args.wandb_project or auto_proj or "isaaclab-async-mbpo"
            algo_name = "AsyncMBPO"
            safe_task = args.task.replace(":", "_").replace("/", "_")
            ts_run = time.strftime("%Y%m%d_%H%M%S")
            run_name = args.wandb_run_name or f"{algo_name}_{safe_task}_{ts_run}"
            cfg_dict = {
                "env/task": args.task,
                "env/num_envs": env_cfg.scene.num_envs,
                "env/device": args.device,
                "env/seed": args.seed,
                "algo/name": "AsyncMBPO",
                "algo/gamma": config['gamma'],
                "algo/tau": config['tau'],
                "algo/alpha": config['alpha'],
                "algo/automatic_entropy_tuning": config['automatic_entropy_tuning'],
                "network/policy_hidden_dims": config['policy_hidden_dims'],
                "network/critic_hidden_dims": config['critic_hidden_dims'],
                "network/dynamics_hidden_dims": config['dynamics_hidden_dims'],
                "train/num_epoch": config['num_epoch'],
                "train/epoch_length": config['epoch_length'],
                "train/batch_size": config['policy_train_batch_size'],
                "train/init_exploration_steps": config['init_exploration_steps'],
                "mbpo/num_networks": config['num_networks'],
                "mbpo/num_elites": config['num_elites'],
                "mbpo/rollout_batch_size": config['rollout_batch_size'],
                "mbpo/rollout_min_length": config['rollout_min_length'],
                "mbpo/rollout_max_length": config['rollout_max_length'],
                "mbpo/real_ratio": config['real_ratio'],
                "buffer/replay_size": config['replay_size'],
                "buffer/min_pool_size": config['min_pool_size'],
                "lr/unified": config['lr'],
                "lr/critic": config.get('critic_lr', config['lr']),
                "lr/policy": config.get('policy_lr', config['lr']),
                "lr/alpha": config.get('alpha_lr', config['lr']),
                "lr/dynamics": config.get('dynamics_lr', config['lr']),
                # Async training specific
                "async/num_collectors": config['num_collectors'],
                "async/max_queue_size": config['max_queue_size'],
                "async/collection_batch_size": config['collection_batch_size'],
            }
            wandb_run = wandb.init(project=project_name, name=run_name, config=cfg_dict)
        except Exception as _wandb_err:  # noqa: F841
            wandb_run = None

    # SAC agent - use unified config
    class SACArgs:
        def __init__(self, config, device):
            self.gamma = config['gamma']
            self.tau = config['tau']
            self.alpha = config['alpha']
            self.policy = "Gaussian"
            self.target_update_interval = config['target_update_interval']
            self.automatic_entropy_tuning = config['automatic_entropy_tuning']
            self.cuda = device.startswith("cuda")
            self.lr = config['lr']
            self.critic_lr = config['critic_lr']
            self.policy_lr = config['policy_lr']
            self.alpha_lr = config['alpha_lr']
            self.policy_hidden_dims = config['policy_hidden_dims']
            self.critic_hidden_dims = config['critic_hidden_dims']
            self.num_epoch = config['num_epoch']
            self.epoch_length = config['epoch_length']

    sac_args = SACArgs(config, args.device)

    class _ActSpace:
        def __init__(self, dim):
            self.shape = (dim,)
            self.high = np.ones(dim, dtype=np.float32)
            self.low = -np.ones(dim, dtype=np.float32)

    agent = SAC(obs_dim, _ActSpace(act_dim), sac_args)

    # Online Incremental Dynamics (ensemble)
    # parse hidden dims from CLI string override if present
    try:
        hidden_override = [int(x) for x in getattr(args, 'dynamics_hidden_dims', '256,256').split(',') if x.strip()]
        if len(hidden_override) == 0:
            hidden_override = config['dynamics_hidden_dims']
    except Exception:
        hidden_override = config['dynamics_hidden_dims']
    
    # Create ensemble of dynamics models
    num_networks = config['num_networks']
    num_elites = config['num_elites']
    dynamics_models = []
    dynamics_optimizers = []
    for i in range(num_networks):
        model = IncrementalDynamicsModel(obs_dim, act_dim, hidden_override).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('dynamics_lr', config['lr']))
        dynamics_models.append(model)
        dynamics_optimizers.append(optimizer)
    
    # Elite selection tracking
    elite_indices = list(range(num_elites))  # Start with first N models as elites
    model_losses = [float('inf')] * num_networks  # Track validation losses

    # Create async trainer
    async_trainer = create_async_trainer(
        vec_env=vec_env,
        agent=agent,
        dynamics_models=dynamics_models,
        dynamics_optimizers=dynamics_optimizers,
        config=config,
        device=args.device
    )

    # Prepare checkpoint dir
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_task = args.task.replace(":", "_").replace("/", "_")
    algo_name = "AsyncMBPO"
    task_dir = os.path.join(args.ckpt_dir, safe_task)
    run_ckpt_dir = os.path.join(task_dir, f"{algo_name}_{ts}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    print(f"[Checkpoint] directory: {run_ckpt_dir}")

    # Start async training
    print("[AsyncMBPO] Starting asynchronous training...")
    async_trainer.start_training()

    # Main monitoring loop
    total_step = 0
    log_interval = config['log_interval']
    ckpt_interval = config['ckpt_interval']
    eval_interval = config['eval_interval']
    wall_start_time = time.time()
    last_log_step = -1

    try:
        while total_step < config['num_steps']:
            time.sleep(1.0)  # Check every second
            
            # Get training statistics
            stats = async_trainer.get_training_stats()
            
            # Log training progress
            if (total_step - last_log_step) >= log_interval:
                last_log_step = total_step
                elapsed = max(1e-6, time.time() - wall_start_time)
                
                print(f"[AsyncMBPO] step={total_step} "
                      f"real_data={stats['real_data_collected']} "
                      f"model_data={stats['model_data_collected']} "
                      f"agent_updates={stats['agent_updates']} "
                      f"dynamics_updates={stats['dynamics_updates']} "
                      f"real_buffer={stats['real_buffer_size']} "
                      f"model_buffer={stats['model_buffer_size']} "
                      f"real_queue={stats['real_queue_size']} "
                      f"model_queue={stats['model_queue_size']}")
                
                if wandb_run is not None:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "Train/step": total_step,
                            "Async/real_data_collected": stats['real_data_collected'],
                            "Async/model_data_collected": stats['model_data_collected'],
                            "Async/agent_updates": stats['agent_updates'],
                            "Async/dynamics_updates": stats['dynamics_updates'],
                            "Buffers/real_buffer_size": stats['real_buffer_size'],
                            "Buffers/model_buffer_size": stats['model_buffer_size'],
                            "Buffers/real_queue_size": stats['real_queue_size'],
                            "Buffers/model_queue_size": stats['model_queue_size'],
                            "Perf/total_fps": total_step / elapsed,
                        }, step=total_step)
                    except Exception:
                        pass
            
            # Periodic checkpoint
            if total_step % max(1, ckpt_interval) == 0:
                try:
                    ckpt = {
                        "epoch": int(total_step // max(1, config['epoch_length'])),
                        "total_step": total_step,
                        "obs_dim": obs_dim,
                        "act_dim": act_dim,
                        "sac": {
                            "critic": agent.critic.state_dict(),
                            "critic_target": agent.critic_target.state_dict(),
                            "policy": agent.policy.state_dict(),
                            "critic_optim": agent.critic_optim.state_dict(),
                            "policy_optim": agent.policy_optim.state_dict(),
                            "alpha": float(agent.alpha if isinstance(agent.alpha, torch.Tensor) else agent.alpha),
                            "alpha_optim": None,
                        },
                        "env_model": {
                            "models": [model.state_dict() for model in dynamics_models],
                            "optimizers": [optim.state_dict() for optim in dynamics_optimizers],
                            "elite_indices": elite_indices,
                            "model_losses": model_losses,
                        },
                        "cfg": {
                            "task": args.task,
                            "num_envs": vec_env.num_envs,
                            "device": args.device,
                        },
                        "async_stats": stats,
                    }
                    ckpt_path = os.path.join(run_ckpt_dir, f"ckpt_step_{total_step}.pt")
                    torch.save(ckpt, ckpt_path)
                    with open(os.path.join(run_ckpt_dir, "latest.json"), "w") as f:
                        json.dump({"latest": ckpt_path, "epoch": int(total_step // max(1, config['epoch_length'])), "total_step": total_step}, f)
                    print(f"[Checkpoint] saved at {ckpt_path}")
                    if wandb_run is not None:
                        try:
                            import wandb  # type: ignore
                            wandb.save(ckpt_path)
                        except Exception:
                            pass
                except Exception:
                    pass
            
            total_step += 1

    except KeyboardInterrupt:
        print("\n[AsyncMBPO] Training interrupted by user")

    # Stop async training
    print("[AsyncMBPO] Stopping asynchronous training...")
    async_trainer.stop_training()

    print("[Done] training finished. Closing env & app.")
    # Close vec env
    try:
        if 'vec_env' in locals() and vec_env is not None:
            vec_env.close()
    except Exception:
        pass
    
    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass
    simulation_app.close()


if __name__ == "__main__":
    main()
