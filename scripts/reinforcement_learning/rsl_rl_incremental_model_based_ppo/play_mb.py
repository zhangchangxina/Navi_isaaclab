# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL (Incremental Model-Based PPO)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

# local imports
# Ensure we can import from the directory containing this script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with MB-PPO.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch

# MB-PPO Runner
from mb_on_policy_runner import MbOnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # Using the MB-PPO specific config parser
    # Note: MbOnPolicyRunner takes a dict, so we will use cli_args.parse_rsl_rl_cfg result and convert
    # But MB runner might expect additional args that are in the config dict.
    # The MB runner config structure is: { "train_cfg": ... }
    # However, parse_rsl_rl_cfg returns a dataclass or dict suitable for OnPolicyRunner.
    # We will manually construct the config structure expected by MbOnPolicyRunner if needed,
    # or rely on the fact that MbOnPolicyRunner mirrors OnPolicyRunner args + extras.
    
    # We load the config as a dictionary directly from the yaml + CLI overrides
    # But cli_args.parse_rsl_rl_cfg returns a config object. 
    # Let's use the standard utility and convert to dict.
    agent_cfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    
    # Force add cli args to the config dict for MB-specific params
    agent_cfg_dict = agent_cfg.to_dict()
    # mb_on_policy_runner expects these in the root of the config or handled via kw_args
    # We will pass them as kwargs to the runner init.
    
    # Check if use_incremental_actions is set
    use_incremental_actions = getattr(args_cli, "use_incremental_actions", False)
    use_cbf = getattr(args_cli, "use_cbf", False)
    cbf_gamma = getattr(args_cli, "cbf_gamma", 0.5)
    
    print(f"[INFO] Incremental Actions Mode: {use_incremental_actions}")
    print(f"[INFO] CBF Safety Filter: {use_cbf} (gamma={cbf_gamma})")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # load previously trained model using MB runner
    # We pass the MB-specific arguments from CLI if they exist
    dynamics_hidden_dims = getattr(args_cli, "dynamics_hidden_dims", [512, 512])
    mb_kwargs = {
        "use_incremental_actions": use_incremental_actions,
        "use_cbf": use_cbf,
        "cbf_gamma": cbf_gamma,
        "dynamics_hidden_dims": dynamics_hidden_dims,
    }
    
    # Construct runner
    ppo_runner = MbOnPolicyRunner(env, agent_cfg_dict, log_dir=None, device=agent_cfg.device, **mb_kwargs)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.policy
    policy.eval()

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    
    # Init incremental action accumulator
    num_envs = env.num_envs
    num_actions = env.num_actions
    accum_actions = torch.zeros((num_envs, num_actions), device=env.device)
    # Track previous actions for absolute mode + CBF
    prev_actions = torch.zeros((num_envs, num_actions), device=env.device)
    
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            # MbOnPolicyRunner normally handles normalization in its learn() loop.
            # Here we must manually normalize if needed.
            # The runner has obs_normalizer.
            obs_norm = ppo_runner.obs_normalizer(obs)
            
            # Policy inference
            # MB runner policies often expect normalized obs
            actions = policy.act_inference(obs_norm)
            actions_tensor = actions # alias
            
            # --- Apply CBF correction if enabled (Must match MB runner logic) ---
            # NOTE: MbOnPolicyRunner handles CBF inside its `learn` loop, but not automatically in `play_mb.py`
            # We need to manually invoke CBF here if we want to visualize it.
            if use_cbf and ppo_runner._cbf is not None and ppo_runner._dyn_models:
                # Need real obs (raw) for CBF
                # obs is currently raw (get_observations returns raw)
                # ppo_runner.lidar_dim is set
                
                # We need dynamics model (use the first one)
                dyn_model = ppo_runner._dyn_models[0]
                
                # We need to prepare actions for CBF (which expects delta if incremental, or nominal delta if absolute)
                if use_incremental_actions:
                    d_real = actions_tensor
                    # Solve CBF
                    d_safe = ppo_runner._cbf.solve_cbf_qp(dyn_model, obs, d_real)
                    # Replace action
                    actions = d_safe
                else:
                    # Absolute mode: we need to compute nominal delta
                    # d_nom = a_real - prev_actions
                    d_nom = actions_tensor - prev_actions
                    
                    # Solve CBF for delta
                    d_safe = ppo_runner._cbf.solve_cbf_qp(dyn_model, obs, d_nom)
                    
                    # Reconstruct safe absolute action
                    # a_safe = prev + d_safe
                    actions = prev_actions + d_safe
            
            # Handle incremental actions
            if use_incremental_actions:
                # actions are delta
                accum_actions += actions
                actions_to_env = accum_actions.clone()
            else:
                actions_to_env = actions
            
            # env stepping
            obs, _, dones, _ = env.step(actions_to_env)
            
            # Reset accumulators for done envs
            if use_incremental_actions:
                reset_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    accum_actions[reset_ids] = 0.0
            else:
                # Update prev_actions for next step (Absolute mode)
                prev_actions = actions_to_env.detach().clone()
                reset_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0:
                    prev_actions[reset_ids] = 0.0

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

