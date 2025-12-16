# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass


@configclass
class DroneSACCfg:
    """SAC defaults for UAV (Drone), aligned in scale with PPO configs.

    These are consumed by external scripts (e.g., scripts/incremental_mbpo/train_isaac_sac.py).
    They do not directly change environment behavior.
    """

    num_envs: int = 256
    num_steps_per_env: int = 8
    max_iterations: int = 2000
    log_interval: int = 1  # Increased for better performance
    eval_interval: int = 100  # Increased for better performance
    ckpt_interval: int = 500
    
    # core
    policy_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True




    batch_size: int = 256
    updates_per_cycle: int = 1000

    # MBPO-specific defaults (merged here for a single task-side cfg)
    # ensemble dynamics
    num_networks: int = 1  # Minimal ensemble for more environments
    num_elites: int = 1
    # 动力学模型网络结构
    dynamics_hidden_dims: list[int] = [512, 256, 512]
    # UTD：每个并行环境每步的更新次数系数
    utd: int = 1
    # utd_critic_per_env: int = 1
    # utd_actor_per_env: int = 1
    # utd_dynamic_per_env: int = 1  # Reduced from 4 for better performance

    # rollout schedule
    rollout_batch_size: int = 256  # Further reduced for better performance
    rollout_min_length: int = 0
    rollout_max_length: int = 0   # Further reduced for better performance


    # buffers & training for MBPO
    replay_size: int = 1_000_000
    min_pool_size: int = 10000
    real_ratio: float = 1
    # Linear decay schedule for real_ratio (by iterations): from real_ratio -> real_ratio_min over ~max_iterations
    real_ratio_min: float = 1

    
    init_exploration_steps: int = 0
    
    # training control
    target_update_interval: int = 1
   
    

    
    # Stability reward & control shaping
    use_stability_reward: bool = False
    Q_scale: float = 1.0
    stability_coef: float = 1e-3
    use_cumulative_action: bool = False
    
    # LQR baseline (optional, uses offline incremental dynamics to estimate B(x))
    use_lqr_baseline: bool = True
    lqr_model_path: str | None = None
    formulation: str = "f2"  # choices: ["f1", "f2"]
    R_scale: float = 0.1
    
    # Incremental dynamics online training hyper-params
    dyn_w_dyn: float = 1.0
    dyn_w_r: float = 1.0
    dyn_w_d: float = 0.1
    dyn_done_threshold: float = 0.5
    
    # RSL-RL style update parameters
    use_rsl_rl_updates: bool = False  # Enable RSL-RL style updates by default
    num_learning_epochs: int = 5
    num_mini_batches: int = 4

    # notes
    notes: str = (
        "Use with: ./isaaclab.sh -p scripts/incremental_mbpo/train_isaac_sac.py "
        "--task=Isaac-Exploration-Rough-Drone-v0 --device=cuda:0 --headless --profile uav. "
        "Note: current SAC code uses single hidden_size; lists are provided for parity with PPO configs."
    )


