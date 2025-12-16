import argparse
import sys
import os
import numpy as np
import time
from typing import Optional
import json
import os
import torch
from scipy import linalg

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


def parse_args():
    parser = argparse.ArgumentParser(description="Train MBPO (v2) on Isaac task with stability reward")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel Isaac sub-environments")
    # checkpointing
    parser.add_argument("--ckpt_dir", type=str, default="runs/mbpo_ckpts", help="Directory to save checkpoints")
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
    # Stability reward & control shaping
    parser.add_argument("--use_stability_reward", action="store_true", help="Enable Lyapunov stability reward shaping")
    parser.add_argument("--Q_scale", type=float, default=1.0, help="State cost scaling for Lyapunov P = Q_scale * I")
    parser.add_argument("--stability_coef", type=float, default=1e-3, help="Scaling for stability reward term")
    parser.add_argument("--use_cumulative_action", action="store_true", help="Use cumulative action = sum of delta actions per episode")
    # LQR baseline (optional, uses offline incremental dynamics to estimate B(x))
    parser.add_argument("--use_lqr_baseline", action="store_true", help="Use LQR-based Lyapunov P from linearization B(x)")
    parser.add_argument("--lqr_model_path", type=str, default=None, help="Path to offline incremental dynamics checkpoint (.pt)")
    parser.add_argument("--formulation", type=str, default="f2", choices=["f1", "f2"], help="Incremental dynamics formulation for B(x)")
    parser.add_argument("--R_scale", type=float, default=0.1, help="Control cost scaling for LQR (R = R_scale * I)")
    # Incremental dynamics online training hyper-params
    parser.add_argument("--dynamics_hidden_dims", type=str, default="256,256", help="Comma-separated hidden dims for incremental dynamics, e.g., 512,256,128")
    parser.add_argument("--dyn_w_dyn", type=float, default=1.0, help="Weight for dynamics loss term")
    parser.add_argument("--dyn_w_r", type=float, default=1.0, help="Weight for reward loss term")
    parser.add_argument("--dyn_w_d", type=float, default=0.1, help="Weight for done BCE loss term")
    parser.add_argument("--dyn_done_threshold", type=float, default=0.5, help="Done probability threshold during model rollout")
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


def set_rollout_length_by_step(total_step: int, config: dict):
    frac = float(total_step) / max(1, int(config.get('num_steps', 1)))
    frac = float(np.clip(frac, 0.0, 1.0))
    return int(config['rollout_min_length'] + frac * (config['rollout_max_length'] - config['rollout_min_length']))


def construct_lyapunov(device: torch.device, x: np.ndarray | torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x_t = x.to(device=device, dtype=torch.float32)
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
    XP = torch.matmul(x_t, P)
    V = (XP * x_t).sum(dim=-1)
    return V


def _solve_lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    P = linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    return P, K

# (duplicate removed)


def _estimate_B_from_online_model(inc_model: torch.nn.Module, obs_dim: int, act_dim: int, device: torch.device, epsilon: float = 1e-3) -> np.ndarray:
    inc_model.eval()
    with torch.no_grad():
        s0 = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
        a0 = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
        ns0, _, _ = inc_model(s0, a0)
        B_cols = []
        for j in range(act_dim):
            a = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
            a[0, j] = epsilon
            ns_j, _, _ = inc_model(s0, a)
            col = (ns_j - ns0).view(-1).detach().cpu().numpy() / float(epsilon)
            B_cols.append(col)
        B = np.stack(B_cols, axis=1)
    return B


def _setup_lqr_controller(device: torch.device, obs_dim: int, act_dim: int, args) -> tuple[Optional[torch.Tensor], Optional[np.ndarray]]:
    """Build incremental dynamics B(s), solve LQR for P, K at desired state, return P as tensor and K as numpy.

    Falls back to identity P if anything fails.
    """
    if not getattr(args, "use_lqr_baseline", False):
        return None, None
    if getattr(args, "lqr_model_path", None) in (None, ""):
        return None, None
    try:
        # auto-detect model format
        ckpt = torch.load(args.lqr_model_path, map_location=device)
        if isinstance(ckpt, dict) and ("model" in ckpt):
            # 增量动力学（离线）：从 checkpoint 的 'model' 解析 B(s)
            class IncrementalB(torch.nn.Module):
                def __init__(self, s_dim, u_dim):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(s_dim + u_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, s_dim * u_dim),
                    )
                def forward(self, s, du):
                    sa = torch.cat([s, du], dim=-1)
                    return self.net(sa)
            demo = IncrementalB(obs_dim, act_dim).to(device)
            state_dict = ckpt.get("model", ckpt)
            demo.load_state_dict(state_dict)
            demo.eval()
            with torch.no_grad():
                x_d = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
                du_zero = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
                B_flat = demo(x_d, du_zero)
                B = B_flat.view(obs_dim, act_dim).detach().cpu().numpy()
        else:
            # 兼容：按增量动力学 MLP 结构尝试加载
            class IncrementalB2(torch.nn.Module):
                def __init__(self, s_dim, u_dim):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(s_dim + u_dim, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, 256),
                        torch.nn.ReLU(),
                        torch.nn.Linear(256, s_dim * u_dim),
                    )
                def forward(self, s, du):
                    sa = torch.cat([s, du], dim=-1)
                    return self.net(sa)
            demo = IncrementalB2(obs_dim, act_dim).to(device)
            state_dict = ckpt.get("model", ckpt)
            demo.load_state_dict(state_dict)
            demo.eval()
            with torch.no_grad():
                x_d = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
                du_zero = torch.zeros(1, act_dim, dtype=torch.float32, device=device)
                B_flat = demo(x_d, du_zero)
                B = B_flat.view(obs_dim, act_dim).detach().cpu().numpy()
        A = (1.0 - 1e-3) * np.eye(obs_dim)
        Q = float(getattr(args, "Q_scale", 1.0)) * np.eye(obs_dim)
        R = float(getattr(args, "R_scale", 0.1)) * np.eye(act_dim)
        P_np, K = _solve_lqr(A, B, Q, R)
        P_tensor = torch.tensor(P_np, dtype=torch.float32, device=device)
        return P_tensor, K
    except Exception:
        return None, None


def main():
    args = parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Import Isaac task registry and helpers AFTER launching the app
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    from scripts.incremental_mbpo.isaac_adapter import IsaacSingleEnvAdapter, IsaacVecEnvAdapter
    from scripts.incremental_mbpo.isaac_mbpo import IsaacVecSampler

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

    # SAC/MBPO agent defaults (read unified task cfg)
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
            # Allow removal from cfg; env auto-schedule will override epoch_length
            'epoch_length': getattr(task_cfg, 'epoch_length', 1000),
            'num_epoch': getattr(task_cfg, 'num_epoch', 1000),
            'max_path_length': 1000,  # Default fallback, will be overridden by env_cfg
            # incremental dynamics extra (optional in cfg)
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
            # incremental dynamics extras from CLI override
            try:
                if hasattr(args, 'dynamics_hidden_dims') and isinstance(args.dynamics_hidden_dims, str):
                    cli_hidden = [int(x) for x in args.dynamics_hidden_dims.split(',') if x.strip()]
                    if len(cli_hidden) > 0:
                        config['dynamics_hidden_dims'] = cli_hidden
            except Exception:
                pass
            if hasattr(args, 'dyn_w_dyn'):
                config['dyn_w_dyn'] = float(args.dyn_w_dyn)
            if hasattr(args, 'dyn_w_r'):
                config['dyn_w_r'] = float(args.dyn_w_r)
            if hasattr(args, 'dyn_w_d'):
                config['dyn_w_d'] = float(args.dyn_w_d)
            if hasattr(args, 'dyn_done_threshold'):
                config['dyn_done_threshold'] = float(args.dyn_done_threshold)
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
            project_name = args.wandb_project or auto_proj or "isaaclab-mbpo"
            algo_name = "MBPO"
            safe_task = args.task.replace(":", "_").replace("/", "_")
            ts_run = time.strftime("%Y%m%d_%H%M%S")
            run_name = args.wandb_run_name or f"{algo_name}_{safe_task}_{ts_run}"
            cfg_dict = {
                "env/task": args.task,
                "env/num_envs": env_cfg.scene.num_envs,
                "env/device": args.device,
                "env/seed": args.seed,
                "algo/name": "MBPO",
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
                # v2 extras
                "stability/use": bool(getattr(args, "use_stability_reward", False)),
                "stability/Q_scale": float(getattr(args, "Q_scale", 1.0)),
                "stability/coef": float(getattr(args, "stability_coef", 1e-3)),
                "control/use_cumulative": bool(getattr(args, "use_cumulative_action", False)),
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
    from scripts.incremental_mbpo.algo.incremental_dynamics import IncrementalDynamicsModel, incremental_dynamics_loss
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

    env_pool = ReplayMemory(config['replay_size'])
    model_pool = ReplayMemory(config['replay_size'])
    sampler = IsaacVecSampler(vec_env, max_path_length=config['max_path_length'])
    # Separate eval env/sampler to avoid interfering with training state
    try:
        eval_env = IsaacVecEnvAdapter(args.task, env_cfg=env_cfg, device=args.device, render=False)
        eval_sampler = IsaacVecSampler(eval_env, max_path_length=config['max_path_length'])
    except Exception:
        eval_env = None
        eval_sampler = None

    # Prepare checkpoint dir as: <ckpt_dir>/<task>/<ALG>_<timestamp>
    ts = time.strftime("%Y%m%d_%H%M%S")
    safe_task = args.task.replace(":", "_").replace("/", "_")
    algo_name = "MBPO"
    task_dir = os.path.join(args.ckpt_dir, safe_task)
    run_ckpt_dir = os.path.join(task_dir, f"{algo_name}_{ts}")
    os.makedirs(run_ckpt_dir, exist_ok=True)
    print(f"[Checkpoint] directory: {run_ckpt_dir}")

    # Stability reward setup
    device = torch.device(args.device)
    P_tensor = None
    K_lqr = None
    if getattr(args, "use_stability_reward", False):
        # Try LQR baseline first if requested (offline)
        if getattr(args, "use_lqr_baseline", False) and getattr(args, "lqr_model_path", None):
            P_tensor, K_lqr = _setup_lqr_controller(device, obs_dim, act_dim, args)
        # Initialize with diagonal P; if no offline model, we will estimate B from online model later
        if P_tensor is None:
            P_tensor = torch.eye(obs_dim, dtype=torch.float32, device=device) * float(getattr(args, "Q_scale", 1.0))

    # Exploration
    steps = 0
    wall_start_time = time.time()
    recent_rew_sum = 0.0
    recent_rew_count = 0
    # cumulative actions buffer (vectorized)
    cum_action = np.zeros((vec_env.num_envs, act_dim), dtype=np.float32)
    while steps < config['init_exploration_steps']:
        s, a_delta, ns, r, d, info = sampler.sample(agent)
        if getattr(args, "use_cumulative_action", False):
            a = cum_action + a_delta
            cum_action = a.copy()
        else:
            a = a_delta
        # stability reward shaping (exploration)
        shaped_r = r.copy()
        if getattr(args, "use_stability_reward", False) and P_tensor is not None:
            V_current = construct_lyapunov(device, s, P_tensor).detach().cpu().numpy()
            V_next = construct_lyapunov(device, ns, P_tensor).detach().cpu().numpy()
            stability_terms = -float(args.stability_coef) * (V_next - V_current)
            shaped_r = r + stability_terms
        for i in range(vec_env.num_envs):
            env_pool.push(s[i], a[i], shaped_r[i], ns[i], d[i])
            recent_rew_sum += float(shaped_r[i])
            recent_rew_count += 1
            steps += 1
            if d[i]:
                cum_action[i] = 0.0
            if steps >= config['init_exploration_steps']:
                break
        if steps % 1000 == 0 or steps == config['init_exploration_steps']:
            elapsed = max(1e-6, time.time() - wall_start_time)
            env_steps = steps
            print(
                f"[Exploration] collected={steps}/{config['init_exploration_steps']} env_pool={len(env_pool)} "
                f"avg_rew_recent={recent_rew_sum/max(1,recent_rew_count):.3f} throughput_env_s={env_steps/elapsed:.1f}"
            )
            if wandb_run is not None:
                try:
                    import wandb  # type: ignore
                    log_dict = {
                        "Exploration/collected_steps": steps,
                        "Exploration/target_steps": config['init_exploration_steps'],
                        "Exploration/progress": steps / config['init_exploration_steps'],
                        "Buffers/env_pool_size": len(env_pool),
                        "Buffers/model_pool_size": 0,
                        "Train/mean_reward": recent_rew_sum / max(1, recent_rew_count),
                        "Perf/collection_time": elapsed,
                        "Perf/total_fps": env_steps / elapsed,
                        "System/num_envs": vec_env.num_envs,
                    }
                    if getattr(args, "use_stability_reward", False) and P_tensor is not None:
                        log_dict.update({
                            "Stability/use": 1.0,
                            "Stability/Q_scale": float(args.Q_scale),
                            "Stability/coef": float(args.stability_coef),
                        })
                    wandb.log(log_dict, step=steps)
                except Exception:
                    pass

    total_step = 0
    rollout_length = 1
    last_log_step = -1
    log_interval = config['log_interval']
    cum_action = np.zeros((vec_env.num_envs, act_dim), dtype=np.float32)
    # Drive training solely by num_steps but preserve epoch-aligned scheduling/logging
    while total_step < config['num_steps']:
        epoch = int(total_step // max(1, config['epoch_length']))
        # Train ensemble models every step (like critic training)
        if (total_step > 0) and (config['real_ratio'] < 1.0) and len(env_pool) > config['min_pool_size']:
            # Train all dynamics models with small batch from env buffer
            s_batch, a_batch, r_batch, ns_batch, d_batch = env_pool.sample(256)  # Use fixed batch size
            if len(s_batch) > 0:
                s_t = torch.tensor(s_batch, dtype=torch.float32, device=args.device)
                a_t = torch.tensor(a_batch, dtype=torch.float32, device=args.device)
                r_t = torch.tensor(r_batch.reshape(-1), dtype=torch.float32, device=args.device)
                ns_t = torch.tensor(ns_batch, dtype=torch.float32, device=args.device)
                d_t = torch.tensor(d_batch.astype(np.float32).reshape(-1), dtype=torch.float32, device=args.device)
                
                # Train each model in the ensemble with UTD (optimized)
                dynamic_updates = max(1, vec_env.num_envs * int(getattr(args, 'utd_dynamic_per_env', 4)))
                for update_step in range(dynamic_updates):
                    # Batch training for better performance
                    for i, (model, optimizer) in enumerate(zip(dynamics_models, dynamics_optimizers)):
                        loss = incremental_dynamics_loss(
                            model,
                            s_t,
                            a_t,
                            r_t,
                            ns_t,
                            d_t,
                            w_dyn=float(config.get('dyn_w_dyn', 1.0)),
                            w_r=float(config.get('dyn_w_r', 1.0)),
                            w_d=float(config.get('dyn_w_d', 0.1)),
                        )
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        # Track loss for elite selection (simple moving average)
                        if model_losses[i] == float('inf'):
                            model_losses[i] = loss.item()  # Initialize with first loss
                        else:
                            model_losses[i] = 0.9 * model_losses[i] + 0.1 * loss.item()
                
                # Elite selection every 500 steps (reduced frequency for performance)
                if total_step % 500 == 0:
                    # Sort models by loss and select elites
                    sorted_indices = sorted(range(num_networks), key=lambda i: model_losses[i])
                    elite_indices = sorted_indices[:num_elites]
                    elite_losses = [f'{model_losses[i]:.4f}' for i in elite_indices]
                    print(f"[EliteSelection] step={total_step} elite_models={elite_indices} losses={elite_losses}")
                
                # If no offline LQR and stability is enabled, estimate B from best elite model and refresh P,K
                if getattr(args, "use_stability_reward", False) and not getattr(args, "lqr_model_path", None):
                    try:
                        # Use the best elite model for B estimation
                        best_elite_model = dynamics_models[elite_indices[0]]
                        B_est = _estimate_B_from_online_model(best_elite_model, obs_dim, act_dim, device)
                        A = (1.0 - 1e-3) * np.eye(obs_dim)
                        Q = float(getattr(args, "Q_scale", 1.0)) * np.eye(obs_dim)
                        R = float(getattr(args, "R_scale", 0.1)) * np.eye(act_dim)
                        P_np, K_lqr = _solve_lqr(A, B_est, Q, R)
                        P_tensor = torch.tensor(P_np, dtype=torch.float32, device=device)
                    except Exception:
                        pass
                
            # Model rollout every 2 steps (reduced frequency for performance)
            if (total_step % 2 == 0) and len(model_pool) < config['replay_size'] * 0.5:  # Only rollout if model pool is not too full
                rl = set_rollout_length_by_step(total_step, config)
                if rl != rollout_length:
                    rollout_length = rl
                    # Only print rollout length changes every log_interval steps for performance
                    if (total_step - last_log_step) >= log_interval:
                        print(f"[MBPO] epoch={epoch} step={total_step} set rollout_length={rollout_length}")

                states, _, _, _, _ = env_pool.sample_all_batch(config['rollout_batch_size'])
                # For cumulative control in model rollout
                cum_action_model = np.zeros((states.shape[0], act_dim), dtype=np.float32)
                # Track reward statistics during rollout
                rollout_rewards = []
                generated_samples = 0
                for t in range(rollout_length):
                    # Batch action selection for better performance
                    states_tensor = torch.tensor(states, dtype=torch.float32, device=args.device)
                    with torch.no_grad():
                        actions_delta, _, _ = agent.policy.sample(states_tensor)
                    actions_delta = actions_delta.cpu().numpy()
                    if getattr(args, "use_cumulative_action", False):
                        actions = cum_action_model + actions_delta
                        cum_action_model = actions.copy()
                    else:
                        actions = actions_delta
                    # Model step via ensemble incremental dynamics
                    s_t = torch.tensor(states, dtype=torch.float32, device=args.device)
                    a_t = torch.tensor(actions, dtype=torch.float32, device=args.device)
                    
                    # Use ensemble of elite models for prediction (batch processing)
                    with torch.no_grad():
                        # Collect predictions from all elite models
                        elite_predictions = []
                        for elite_idx in elite_indices:
                            elite_model = dynamics_models[elite_idx]
                            ns_t, r_t, dprob_t = elite_model(s_t, a_t)
                            elite_predictions.append((ns_t, r_t, dprob_t))
                        
                        # Stack and average predictions
                        next_states_stack = torch.stack([pred[0] for pred in elite_predictions], dim=0)
                        rewards_stack = torch.stack([pred[1] for pred in elite_predictions], dim=0)
                        terminals_stack = torch.stack([pred[2] for pred in elite_predictions], dim=0)
                        
                        # Average predictions
                        next_states = torch.mean(next_states_stack, dim=0).cpu().numpy()
                        rewards = torch.mean(rewards_stack, dim=0).cpu().numpy()
                        terminals = (torch.mean(terminals_stack, dim=0).cpu().numpy() > float(config.get('dyn_done_threshold', 0.5))).astype(np.bool_)
                    # Stability shaping in model rollout
                    if getattr(args, "use_stability_reward", False) and P_tensor is not None:
                        Vc = construct_lyapunov(device, states, P_tensor)
                        Vn = construct_lyapunov(device, next_states, P_tensor)
                        rewards = rewards + (-float(args.stability_coef) * (Vn - Vc)).detach().cpu().numpy()
                    # Collect reward statistics
                    rollout_rewards.extend(rewards.tolist())
                    generated_samples += len(rewards)
                    samples = [(states[j], actions[j], rewards[j], next_states[j], terminals[j]) for j in range(states.shape[0])]
                    model_pool.push_batch(samples)
                    nonterm = ~terminals.astype(bool)
                    if nonterm.sum() == 0:
                        break
                    states = next_states[nonterm]
                    if getattr(args, "use_cumulative_action", False):
                        cum_action_model = cum_action_model[nonterm]
                # Calculate reward statistics
                avg_rollout_reward = np.mean(rollout_rewards) if rollout_rewards else 0.0
                std_rollout_reward = np.std(rollout_rewards) if rollout_rewards else 0.0
                # Only print rollout info every log_interval steps for performance
                if (total_step - last_log_step) >= log_interval:
                    print(f"[ModelRollout] epoch={epoch} step={total_step} rollout_len={rollout_length} model_pool={len(model_pool)} "
                          f"avg_reward={avg_rollout_reward:.3f} std_reward={std_rollout_reward:.3f} generated_samples={generated_samples} "
                          f"elite_models={elite_indices}")
                if wandb_run is not None:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "MBPO/rollout_length": rollout_length,
                            "MBPO/avg_rollout_reward": avg_rollout_reward,
                            "MBPO/std_rollout_reward": std_rollout_reward,
                            "MBPO/generated_samples": generated_samples,
                            "Buffers/env_pool_size": len(env_pool),
                            "Buffers/model_pool_size": len(model_pool),
                            "Buffers/real_ratio": config['real_ratio'],
                        }, step=total_step)
                    except Exception:
                        pass

        # Real env step
        s, a_delta, ns, r, d, info = sampler.sample(agent)
        if getattr(args, "use_cumulative_action", False):
            a = cum_action + a_delta
            cum_action = a.copy()
        else:
            a = a_delta
        shaped_r = r.copy()
        if getattr(args, "use_stability_reward", False) and P_tensor is not None:
            V_current = construct_lyapunov(device, s, P_tensor).detach().cpu().numpy()
            V_next = construct_lyapunov(device, ns, P_tensor).detach().cpu().numpy()
            stability_terms = -float(args.stability_coef) * (V_next - V_current)
            shaped_r = r + stability_terms
        # batch insert to env_pool
        env_pool.push_batch_tensors(s, a, shaped_r, ns, d)
        recent_rew_sum += float(np.sum(shaped_r))
        recent_rew_count += int(len(shaped_r))
        # reset cumulative action where done
        if getattr(args, "use_cumulative_action", False):
            done_mask = np.asarray(d).astype(bool)
            cum_action[done_mask] = 0.0

        # Update policy with UTD: critic updates = num_envs*4, actor updates = num_envs
        if len(env_pool) > config['min_pool_size']:
            env_bs = int(config['policy_train_batch_size'] * config['real_ratio'])
            model_bs = config['policy_train_batch_size'] - env_bs
            es, ea, er, ens, ed = env_pool.sample(max(env_bs, 1))
            if model_bs > 0 and len(model_pool) > 0:
                ms, ma, mr, mns, md = model_pool.sample_all_batch(max(model_bs, 1))
                bs = np.concatenate([es, ms], axis=0)
                ba = np.concatenate([ea, ma], axis=0)
                br = np.concatenate([er.reshape(-1, 1), mr.reshape(-1, 1)], axis=0).reshape(-1)
                bns = np.concatenate([ens, mns], axis=0)
                bd = np.concatenate([ed.reshape(-1, 1), md.reshape(-1, 1)], axis=0).reshape(-1)
            else:
                bs, ba, br, bns, bd = es, ea, er, ens, ed

            mask = (~bd.astype(bool)).astype(float)
            mem = (bs, ba, br, bns, mask)
            critic_updates = max(1, vec_env.num_envs * int(getattr(args, 'utd_critic_per_env', 4)))
            actor_updates = max(1, vec_env.num_envs * int(getattr(args, 'utd_actor_per_env', 1)))

            # critic-only updates
            last_c1 = last_c2 = 0.0
            for i_c in range(critic_updates):
                c1, c2 = agent.update_critic_only(mem, config['policy_train_batch_size'], total_step)
                last_c1, last_c2 = c1, c2

            # actor-only updates
            last_pol = last_ent = last_alpha = 0.0
            for i_a in range(actor_updates):
                pol, ent, alpha = agent.update_actor_only(mem, config['policy_train_batch_size'], total_step)
                last_pol, last_ent, last_alpha = pol, ent, alpha

            # For logging compatibility
            c1, c2, pol, ent, alpha = last_c1, last_c2, last_pol, last_ent, last_alpha

            if (total_step - last_log_step) >= log_interval:
                last_log_step = total_step
                elapsed = max(1e-6, time.time() - wall_start_time)
                env_steps = (total_step * vec_env.num_envs)
                avg_rew = recent_rew_sum / max(1, recent_rew_count)
                try:
                    cur_lr = agent.policy_optim.param_groups[0]["lr"]
                except Exception:
                    cur_lr = float('nan')
                print(
                    f"[Train] epoch={epoch} step={total_step} env_steps={env_steps} envs={vec_env.num_envs} rl={rollout_length} "
                    f"env_pool={len(env_pool)} model_pool={len(model_pool)} "
                    f"rew_avg_recent={avg_rew:.3f} c1={c1:.4f} c2={c2:.4f} pol={pol:.4f} ent={ent:.4f} alpha={alpha:.4f} lr={cur_lr:.2e}"
                )
                if wandb_run is not None:
                    try:
                        import wandb  # type: ignore
                        log_dict = {
                            "Train/epoch": epoch,
                            "Train/step": total_step,
                            "Train/env_steps": env_steps,
                            "Train/rollout_length": rollout_length,
                            "Loss/critic1": c1,
                            "Loss/critic2": c2,
                            "Loss/policy": pol,
                            "Loss/entropy": ent,
                            "Policy/alpha": alpha,
                            "Buffers/env_pool_size": len(env_pool),
                            "Buffers/model_pool_size": len(model_pool),
                            "Opt/lr": cur_lr,
                            "Train/mean_reward": avg_rew,
                        }
                        if getattr(args, "use_stability_reward", False) and P_tensor is not None:
                            log_dict.update({
                                "Stability/use": 1.0,
                                "Stability/Q_scale": float(args.Q_scale),
                                "Stability/coef": float(args.stability_coef),
                            })
                        wandb.log(log_dict, step=total_step)
                    except Exception:
                        pass

            # periodic evaluation
            if (config.get("eval_interval", 0) > 0) and (total_step % config["eval_interval"] == 0):
                if eval_sampler is not None:
                    try:
                        eval_steps = max(1, int(config.get('epoch_length', 1000)))
                        eval_sum = 0.0
                        eval_count = 0
                        for _ in range(eval_steps):
                            _, _, _, r_eval, _, _ = eval_sampler.sample(agent, eval_t=True)
                            eval_sum += float(np.sum(r_eval))
                            eval_count += int(r_eval.shape[0])
                        avg_eval_rew = eval_sum / max(1, eval_count)
                        print(f"[Eval] step={total_step} avg_reward_per_env_step={avg_eval_rew:.3f}")
                        if wandb_run is not None:
                            try:
                                import wandb  # type: ignore
                                wandb.log({
                                    "Eval/avg_reward_per_env_step": avg_eval_rew,
                                    "Eval/steps": eval_steps,
                                }, step=total_step)
                            except Exception:
                                pass
                    except Exception:
                        pass

            # periodic checkpoint
            if total_step % max(1, config["ckpt_interval"]) == 0:
                    try:
                        ckpt = {
                            "epoch": epoch,
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
                        }
                        ckpt_path = os.path.join(run_ckpt_dir, f"ckpt_step_{total_step}.pt")
                        torch.save(ckpt, ckpt_path)
                        with open(os.path.join(run_ckpt_dir, "latest.json"), "w") as f:
                            json.dump({"latest": ckpt_path, "epoch": epoch, "total_step": total_step}, f)
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

    print("[Done] training finished. Closing env & app.")
    # Close eval env first
    try:
        if 'eval_env' in locals() and eval_env is not None:
            eval_env.close()
    except Exception:
        pass
    # Close training vec env next
    try:
        if 'vec_env' in locals() and vec_env is not None:
            vec_env.close()
    except Exception:
        pass
    # Drop references and collect to avoid __del__ on partially torn-down objects
    try:
        eval_env = None  # type: ignore
        vec_env = None  # type: ignore
        import gc as _gc
        _gc.collect()
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


