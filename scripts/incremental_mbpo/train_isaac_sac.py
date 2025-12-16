import argparse
import sys
import os
import time
from typing import Optional
import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SRC_DIR = os.path.join(_REPO_ROOT, "source")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from omni.isaac.lab.app import AppLauncher
except Exception:
    from isaaclab.app import AppLauncher

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.incremental_mbpo.algo.replay_memory import ReplayMemory
from scripts.incremental_mbpo.algo.sac import SAC
from scripts.incremental_mbpo.isaac_adapter import IsaacSingleEnvAdapter, IsaacVecEnvAdapter


def parse_args():
    parser = argparse.ArgumentParser(description="Step-driven SAC on Isaac task")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_envs", type=int, default=None)
    # ckpt
    parser.add_argument("--ckpt_dir", type=str, default="runs/sac_ckpts")
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    # LRs override
    parser.add_argument("--critic_lr", type=float, default=None)
    parser.add_argument("--policy_lr", type=float, default=None)
    parser.add_argument("--alpha_lr", type=float, default=None)
    # UTD options (updates per data): factors per environment
    parser.add_argument("--utd_critic_per_env", type=int, default=4, help="Critic updates per environment per step (UTD)")
    parser.add_argument("--utd_actor_per_env", type=int, default=1, help="Actor updates per environment per step (UTD)")
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

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # Load task cfg first
    def _load_task_sac_cfg(task_id: str):
        try:
            if "drone" in task_id.lower() or "uav" in task_id.lower():
                from isaaclab_tasks.manager_based.exploration.velocity.config.drone.agents.mbpo_cfg import (
                    DroneSACCfg as SACCfg,
                )
            else:
                from isaaclab_tasks.manager_based.exploration.velocity.config.turtlebot.agents.mbpo_cfg import (
                    TurtlebotSACCfg as SACCfg,
                )
            return SACCfg()
        except Exception:
            return None

    task_cfg = _load_task_sac_cfg(args.task)
    if task_cfg is None:
        raise RuntimeError("No SAC config for task")

    # Load env cfg
    env_cfg = load_cfg_from_registry(args.task.split(":")[-1], "env_cfg_entry_point")
    num_envs = getattr(args, "num_envs", None)
    if num_envs is None:
        # Try to get from task config
        if task_cfg is not None:
            num_envs = getattr(task_cfg, "num_envs", None)
    env_cfg.scene.num_envs = int(num_envs) if num_envs else 1
    env_cfg.sim.device = args.device
    env_cfg.seed = args.seed
    # Disable vis
    try:
        if getattr(env_cfg.scene, "plane", None) is not None:
            env_cfg.scene.plane.debug_vis = False
        if getattr(env_cfg.scene, "terrain", None) is not None:
            env_cfg.scene.terrain.debug_vis = False
        if hasattr(env_cfg, "viewer"):
            env_cfg.viewer.enable = False
    except Exception:
        pass

    # Build env (vector or single)
    if env_cfg.scene.num_envs > 1:
        env = IsaacVecEnvAdapter(args.task, env_cfg=env_cfg, device=args.device, render=getattr(args, "render", False))
        obs_mat, _ = env.reset(seed=args.seed)
        obs_dim = obs_mat.shape[-1]
        act_dim = env.act_dim
    else:
        env = IsaacSingleEnvAdapter(args.task, env_cfg=env_cfg, device=args.device, render=getattr(args, "render", False))
        obs, _ = env.reset(seed=args.seed)
        obs_dim = obs.shape[-1]
        act_dim = getattr(env, 'act_dim', np.asarray(getattr(env, 'action_space').high).shape[-1] if hasattr(env, 'action_space') else obs_dim)


    # Build unified config (step-driven)
    def get_config(task_cfg, args, env_cfg=None):
        if task_cfg is None:
            raise RuntimeError(f"No configuration found for task")
        config = {
            "gamma": task_cfg.gamma,
            "tau": task_cfg.tau,
            "alpha": task_cfg.alpha,
            "lr": 1e-3,  # Default learning rate (fallback)
            "critic_lr": getattr(task_cfg, 'critic_lr', 1e-3),
            "policy_lr": getattr(task_cfg, 'policy_lr', 1e-3),
            "alpha_lr": getattr(task_cfg, 'alpha_lr', 1e-3),
            "policy_hidden_dims": task_cfg.policy_hidden_dims,
            "critic_hidden_dims": task_cfg.critic_hidden_dims,
            "automatic_entropy_tuning": task_cfg.automatic_entropy_tuning,
            "target_update_interval": getattr(task_cfg, "target_update_interval", 1),
            # total steps purely by config
            "num_steps": getattr(task_cfg, "num_steps", 1_000_000),
            # sac specifics
            "start_steps": getattr(task_cfg, "start_steps", 10000),
            "batch_size": getattr(task_cfg, "batch_size", 256),
        }
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
        return config

    config = get_config(task_cfg, args, env_cfg)

    # wandb
    wandb_run: Optional[object] = None
    if getattr(args, "wandb", False):
        try:
            import wandb  # type: ignore
            tl = args.task.lower()
            auto_proj = "UGV_Navigation" if ("turtlebot" in tl or "ugv" in tl) else (
                "UAV_Navigation" if ("drone" in tl or "uav" in tl) else "isaaclab-sac"
            )
            project_name = args.wandb_project or auto_proj or "isaaclab-sac"
            algo_name = "SAC"
            safe_task = args.task.replace(":", "_").replace("/", "_")
            ts_run = time.strftime("%Y%m%d_%H%M%S")
            run_name = args.wandb_run_name or f"{algo_name}_{safe_task}_{ts_run}"
            wandb_run = wandb.init(project=project_name, name=run_name, config={
                "env/task": args.task,
                "env/num_envs": getattr(env, "num_envs", 1),
                "env/device": args.device,
                "env/seed": args.seed,
                "algo/name": "SAC",
                "algo/gamma": config["gamma"],
                "algo/tau": config["tau"],
                "algo/alpha": config["alpha"],
                "algo/automatic_entropy_tuning": config["automatic_entropy_tuning"],
                "network/policy_hidden_dims": config["policy_hidden_dims"],
                "network/critic_hidden_dims": config["critic_hidden_dims"],
                "train/num_steps": config["num_steps"],
                "train/start_steps": config["start_steps"],
                "train/batch_size": config["batch_size"],
                "lr/unified": config["lr"],
                "lr/critic": config.get("critic_lr", config["lr"]),
                "lr/policy": config.get("policy_lr", config["lr"]),
                "lr/alpha": config.get("alpha_lr", config["lr"]),
            })
        except Exception:
            wandb_run = None

    # SAC agent
    class SACArgs:
        def __init__(self, config, device):
            self.gamma = config['gamma']
            self.tau = config['tau']
            self.alpha = config['alpha']
            self.policy = "Gaussian"
            self.target_update_interval = config.get('target_update_interval', 1)
            self.automatic_entropy_tuning = config['automatic_entropy_tuning']
            self.cuda = device.startswith("cuda")
            self.lr = config['lr']
            self.critic_lr = config.get('critic_lr', config['lr'])
            self.policy_lr = config.get('policy_lr', config['lr'])
            self.alpha_lr = config.get('alpha_lr', config['lr'])
            self.policy_hidden_dims = config['policy_hidden_dims']
            self.critic_hidden_dims = config['critic_hidden_dims']

    class _ActSpace:
        def __init__(self, dim):
            self.shape = (dim,)
            self.high = np.ones(dim, dtype=np.float32)
            self.low = -np.ones(dim, dtype=np.float32)

    sac_args = SACArgs(config, args.device)
    agent = SAC(obs_dim, _ActSpace(act_dim), sac_args)
    memory = ReplayMemory(1_000_000)

    # Step-driven loop
    total_steps = 0
    log_interval = 100
    ckpt_interval = max(1, args.ckpt_interval)
    wall_start = time.time()
    state = None
    if getattr(env, 'num_envs', 1) == 1:
        state, _ = env.reset(seed=args.seed)

    while total_steps < config['num_steps']:
        if getattr(env, 'num_envs', 1) > 1:
            # vector env
            if state is None:
                state, _ = env.reset(seed=args.seed)
                actions = np.stack([agent.select_action(s) for s in state], axis=0)
            else:
                actions = np.stack([agent.select_action(s) for s in state], axis=0)
            next_state, reward, done, info = env.step(actions)
            # batch insert
            memory.push_batch_tensors(state, actions, reward, next_state, done)
            total_steps += int(np.size(reward))
            state = next_state
        else:
            if state is None:
                state, _ = env.reset(seed=args.seed)
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            memory.push(state, action, reward, next_state, done)
            total_steps += 1
            state = next_state if not done else None

        # update with UTD (separate critic/actor loops)
        if len(memory) > config['batch_size'] and total_steps > config['start_steps']:
            critic_updates = max(1, getattr(env, 'num_envs', 1) * int(getattr(args, 'utd_critic_per_env', 4)))
            actor_updates = max(1, getattr(env, 'num_envs', 1) * int(getattr(args, 'utd_actor_per_env', 1)))
            # sample once per update (could resample each iteration if desired)
            s, a, r, ns, d = memory.sample(config['batch_size'])
            mem = (s, a, r, ns, (~d.astype(bool)).astype(float))
            last_c1 = last_c2 = 0.0
            for _ in range(critic_updates):
                c1, c2 = agent.update_critic_only(mem, config['batch_size'], total_steps)
                last_c1, last_c2 = c1, c2
            last_pol = last_ent = last_alpha = 0.0
            for _ in range(actor_updates):
                pol, ent, alpha = agent.update_actor_only(mem, config['batch_size'], total_steps)
                last_pol, last_ent, last_alpha = pol, ent, alpha
            critic_1_loss, critic_2_loss, policy_loss, ent_loss = last_c1, last_c2, last_pol, last_ent
        else:
            critic_1_loss = critic_2_loss = policy_loss = ent_loss = alpha = float('nan')

        # log
        if total_steps % log_interval == 0:
            elapsed = max(1e-6, time.time() - wall_start)
            try:
                cur_lr = agent.policy_optim.param_groups[0]["lr"]
            except Exception:
                cur_lr = config['lr']
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore
                    log_dict = {
                        "Train/step": total_steps,
                        "Loss/value_function": (critic_1_loss + critic_2_loss) / 2.0 if np.isfinite(critic_1_loss) and np.isfinite(critic_2_loss) else float('nan'),
                        "Loss/critic1": critic_1_loss,
                        "Loss/critic2": critic_2_loss,
                        "Loss/policy": policy_loss,
                        "Loss/entropy": ent_loss,
                        "Policy/alpha": alpha,
                        "Buffers/replay_size": len(memory),
                        "Opt/lr": cur_lr,
                        "System/num_envs": getattr(env, 'num_envs', 1),
                        "Perf/total_fps": total_steps / elapsed,
                    }
                    if info is not None:
                        for key, value in info.items():
                            if isinstance(key, str) and key.startswith("Episode_Reward/"):
                                if isinstance(value, np.ndarray):
                                    log_dict[key] = float(np.mean(value))
                                else:
                                    log_dict[key] = float(value)
                    wandb.log(log_dict, step=total_steps)
                except Exception:
                    pass

        # ckpt
        if total_steps % ckpt_interval == 0:
            try:
                # Prepare checkpoint dir as: <ckpt_dir>/<task>/<ALG>_<timestamp>
                ts = time.strftime("%Y%m%d_%H%M%S")
                safe_task = args.task.replace(":", "_").replace("/", "_")
                algo_name = "SAC"
                task_dir = os.path.join(args.ckpt_dir, safe_task)
                run_ckpt_dir = os.path.join(task_dir, f"{algo_name}_{ts}")
                os.makedirs(run_ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(run_ckpt_dir, f"ckpt_step_{total_steps}.pt")
                torch.save({
                    "total_steps": total_steps,
                    "sac": {
                        "critic": agent.critic.state_dict(),
                        "critic_target": agent.critic_target.state_dict(),
                        "policy": agent.policy.state_dict(),
                        "critic_optim": agent.critic_optim.state_dict(),
                        "policy_optim": agent.policy_optim.state_dict(),
                    },
                    "cfg": {
                        "task": args.task,
                        "num_envs": getattr(env, "num_envs", 1),
                        "device": args.device,
                    },
                }, ckpt_path)
                # Save latest.json for easy checkpoint loading
                import json
                with open(os.path.join(run_ckpt_dir, "latest.json"), "w") as f:
                    json.dump({"latest": ckpt_path, "total_steps": total_steps}, f)
                print(f"[Checkpoint] saved at {ckpt_path}")
            except Exception:
                pass

    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.finish()
        except Exception:
            pass
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()


