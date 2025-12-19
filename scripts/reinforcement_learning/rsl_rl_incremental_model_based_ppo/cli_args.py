# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL and model-based PPO arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # RSL-RL argument group (kept identical to scripts/reinforcement_learning/rsl_rl/cli_args.py)
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )

    # Incremental model-based PPO argument group
    mb_group = parser.add_argument_group(
        "model_based", description="Arguments for incremental model-based components."
    )
    # Dynamics ensemble
    mb_group.add_argument("--num_networks", type=int, default=1, help="Number of dynamics models in ensemble.")
    mb_group.add_argument("--dynamics_lr", type=float, default=1e-3, help="Learning rate for dynamics models.")
    mb_group.add_argument(
        "--dynamics_hidden_dims",
        type=int,
        nargs="+",
        default=[512, 512],
        help="Hidden layer sizes for dynamics models.",
    )
    # Loss weights for incremental dynamics
    mb_group.add_argument("--dyn_w_dyn", type=float, default=1.0, help="Weight for next-state prediction loss.")
    mb_group.add_argument("--dyn_w_r", type=float, default=1.0, help="Weight for reward prediction loss.")
    mb_group.add_argument("--dyn_w_d", type=float, default=0.1, help="Weight for done prediction loss.")
    mb_group.add_argument(
        "--dyn_done_threshold", type=float, default=0.5, help="Threshold for modeling terminal probability into done flag."
    )
    # Online training schedule
    mb_group.add_argument(
        "--mb_buffer_size", type=int, default=100000, help="Replay size for online dynamics training buffer."
    )
    mb_group.add_argument(
        "--mb_batch_size", type=int, default=1024, help="Batch size for dynamics updates (vectorized)."
    )
    mb_group.add_argument(
        "--mb_update_every", type=int, default=10, help="Run dynamics updates every N env steps (per vector step)."
    )
    mb_group.add_argument(
        "--mb_update_steps", type=int, default=1, help="Number of gradient steps when dynamics update triggers."
    )
    # Virtual model envs (mix synthetic data into PPO)
    mb_group.add_argument(
        "--mb_virtual_envs", type=int, default=8, help="Number of virtual environments driven by dynamics model."
    )
    # Backward-compatible alias: --mb_init_from_real
    mb_group.add_argument(
        "--mb_init_from_buffer", action="store_true", help="Initialize virtual env states by sampling from real buffer."
    )
    mb_group.add_argument(
        "--mb_virt_steps_per_iter",
        type=int,
        default=4,
        help="Max virtual rollout steps after each env reset (0 means unlimited).",
    )
    mb_group.add_argument(
        "--mb_warmup_iters",
        type=int,
        default=3,
        help="Number of PPO iterations to warm up before using dynamics predictions for virtual rollouts.",
    )
    mb_group.add_argument(
        "--mb_sync_update",
        action="store_true",
        help="Run dynamics updates synchronously (by default updates run asynchronously).",
    )
    # Stability shaping (optional)
    mb_group.add_argument("--use_stability_reward", action="store_true", help="Enable Lyapunov-based reward shaping.")
    mb_group.add_argument("--Q_scale", type=float, default=1.0, help="LQR Q scaling for Lyapunov P estimation.")
    mb_group.add_argument("--R_scale", type=float, default=0.1, help="LQR R scaling for Lyapunov P estimation.")
    mb_group.add_argument(
        "--stability_coef", type=float, default=1e-3, help="Coefficient for stability reward shaping term."
    )
    mb_group.add_argument(
        "--mb_P_update_every", type=int, default=200, help="Update Lyapunov matrix P every N env steps."
    )
    # Action representation mode
    mb_group.add_argument(
        "--use_incremental_actions", 
        action="store_true", 
        default=False, 
        help="Whether to use incremental actions (delta) for both step conversion and dynamics modeling."
    )
    # CBF Safety Filter
    mb_group.add_argument(
        "--use_cbf",
        action="store_true",
        default=False,
        help="Whether to enable Control Barrier Function (CBF) safety filter using learned dynamics.",
    )
    mb_group.add_argument(
        "--cbf_safety_distance",
        type=float,
        default=0.5,
        help="Safety distance in meters for CBF. Lidar distances below this trigger correction.",
    )

    # Checkpoint utilities
    mb_group.add_argument(
        "--dyn_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for loading dynamics models. If provided, dynamics will be loaded and policy load will be skipped.",
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and getattr(args_cli, "log_project_name", None):
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg


