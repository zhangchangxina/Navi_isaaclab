# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL, augmented with incremental model-based reward shaping."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL + incremental model-based shaping.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from mb_on_policy_runner import MbOnPolicyRunner

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent, while applying incremental model-based reward shaping."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl_incremental_model_based_ppo", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    # ensure run_name includes algo tag
    try:
        _algo_tag = "mbppo"
        if getattr(agent_cfg, "run_name", None):
            if _algo_tag not in str(agent_cfg.run_name):
                agent_cfg.run_name = f"{agent_cfg.run_name}_{_algo_tag}"
        else:
            agent_cfg.run_name = _algo_tag
    except Exception:
        pass
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create model-based on-policy runner (replaces wrapper)
    runner = MbOnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        device=agent_cfg.device,
        log_dir=log_dir,
        # MB params from CLI
        mb_virtual_envs=getattr(args_cli, "mb_virtual_envs", 0),
        mb_virt_steps_per_iter=getattr(args_cli, "mb_virt_steps_per_iter", 0),
        mb_warmup_iters=getattr(args_cli, "mb_warmup_iters", 0),
        mb_init_from_buffer=(getattr(args_cli, "mb_init_from_buffer", False) or getattr(args_cli, "mb_init_from_real", False)),
        num_networks=getattr(args_cli, "num_networks", 1),
        dynamics_lr=getattr(args_cli, "dynamics_lr", 1e-3),
        dynamics_hidden_dims=getattr(args_cli, "dynamics_hidden_dims", [256, 256]),
        dyn_w_dyn=getattr(args_cli, "dyn_w_dyn", 1.0),
        dyn_w_r=getattr(args_cli, "dyn_w_r", 1.0),
        dyn_w_d=getattr(args_cli, "dyn_w_d", 0.1),
        dyn_done_threshold=getattr(args_cli, "dyn_done_threshold", 0.5),
        mb_buffer_size=getattr(args_cli, "mb_buffer_size", 100000),
        mb_batch_size=getattr(args_cli, "mb_batch_size", 1024),
        mb_update_every=getattr(args_cli, "mb_update_every", 10),
        mb_update_steps=getattr(args_cli, "mb_update_steps", 1),
        use_stability_reward=getattr(args_cli, "use_stability_reward", False),
        Q_scale=getattr(args_cli, "Q_scale", 1.0),
        R_scale=getattr(args_cli, "R_scale", 0.1),
        stability_coef=getattr(args_cli, "stability_coef", 1e-3),
        use_incremental_actions=getattr(args_cli, "use_incremental_actions", False),
        use_cbf=getattr(args_cli, "use_cbf", False),
        cbf_safety_distance=getattr(args_cli, "cbf_safety_distance", 0.5),
    )
    # optional: load checkpoint into runner if requested
    ckpt_path = None
    # If dyn_checkpoint is provided, default to loading ONLY dynamics and skip policy load
    if getattr(args_cli, "dyn_checkpoint", None):
        dyn_ckpt = args_cli.dyn_checkpoint
        if os.path.isfile(dyn_ckpt):
            print(f"[INFO] Loading dynamics only from: {dyn_ckpt}")
            runner.load_dynamics(dyn_ckpt, load_optimizers=bool(args_cli.resume))
        else:
            print(f"[WARN] Provided dyn_checkpoint does not exist: {dyn_ckpt}")
    else:
        # No dyn checkpoint: proceed with policy checkpoint/resume
        if getattr(args_cli, "checkpoint", None):
            if os.path.isfile(args_cli.checkpoint):
                ckpt_path = args_cli.checkpoint
            else:
                print(f"[WARN] Provided checkpoint does not exist: {args_cli.checkpoint}")
        elif getattr(args_cli, "resume", False):
            # compute resume path from experiment logs
            ckpt_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
            if ckpt_path is not None and not os.path.isfile(ckpt_path):
                print(f"[WARN] Resolved resume checkpoint not found: {ckpt_path}")
                ckpt_path = None
        if ckpt_path:
            print(f"[INFO] Loading checkpoint: {ckpt_path}")
            runner.load(ckpt_path, load_optimizer=bool(args_cli.resume))

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()  # type: ignore[call-arg]
    # close sim app
    simulation_app.close()


