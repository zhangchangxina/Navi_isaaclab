# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple
import types

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
)

from scripts.reinforcement_learning.rsl_rl_incremental_model_based_ppo.incremental_dynamics import (
    IncrementalDynamicsModel,
    incremental_dynamics_loss,
)


class MbOnPolicyRunner:
    """On-policy runner that augments PPO with model-based virtual rollouts.

    This class replaces the need for a gym Wrapper by generating virtual samples
    inside the runner and concatenating them with real samples before passing to PPO.
    """

    def __init__(
        self,
        env: VecEnv,
        train_cfg: Dict[str, Any],
        *,
        device: str,
        log_dir: Optional[str] = None,
        # model-based parameters
        mb_virtual_envs: int = 0,
        mb_virt_steps_per_iter: int = 0,
        mb_warmup_iters: int = 0,
        mb_init_from_buffer: bool = False,
        # incremental dynamics settings
        num_networks: int = 1,
        dynamics_lr: float = 1e-3,
        dynamics_hidden_dims: List[int] | Tuple[int, ...] = (256, 256),
        dyn_w_dyn: float = 1.0,
        dyn_w_r: float = 1.0,
        dyn_w_d: float = 0.1,
        dyn_done_threshold: float = 0.5,
        mb_buffer_size: int = 100000,
        mb_batch_size: int = 1024,
        mb_update_every: int = 10,
        mb_update_steps: int = 1,
        # stability shaping
        use_stability_reward: bool = False,
        Q_scale: float = 1.0,
        R_scale: float = 0.1,
        stability_coef: float = 1e-3,
        use_incremental_actions: bool = False,
        use_cbf: bool = False,
    ) -> None:
        self.cfg = dict(train_cfg)
        self.alg_cfg = dict(train_cfg["algorithm"])  # copy to avoid mutation
        self.policy_cfg = dict(train_cfg["policy"])  # copy to avoid mutation
        self.device = device
        self.env = env
        # logging / bookkeeping (aligned with OnPolicyRunner)
        self.log_dir: Optional[str] = log_dir
        self.writer = None
        self.disable_logs: bool = False  # no distributed handling here
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.current_learning_iteration: int = 0
        self.logger_type: str = str(self.cfg.get("logger", "tensorboard")).lower()

        # resolve dimensions from environment
        obs, extras = self.env.get_observations()
        self.num_obs = int(obs.shape[1])
        # privileged obs (critic)
        if "critic" in extras["observations"]:
            self.privileged_obs_type: Optional[str] = "critic"
            self.num_privileged_obs = int(extras["observations"]["critic"].shape[1])
        else:
            self.privileged_obs_type = None
            self.num_privileged_obs = int(self.num_obs)

        # Try to infer lidar dim from env if possible
        self.lidar_dim = 16 # fallback
        try:
            # specifically for IsaacLab ManagerBasedRLEnv
            # cast env to Any to avoid static analysis errors about VecEnv not having unwrapped
            base_env = getattr(self.env, "unwrapped", None)
            if base_env is not None:
                if hasattr(base_env, "observation_manager"):
                    # Assuming group name is "policy"
                    group_name = "policy"
                    obs_manager = base_env.observation_manager
                    
                    if group_name in obs_manager.active_terms:
                        active_terms = obs_manager.active_terms[group_name]
                        term_dims = obs_manager.group_obs_term_dim[group_name]
                        
                        # Look for "lidar_scan"
                        if "lidar_scan" in active_terms:
                            idx = active_terms.index("lidar_scan")
                            # term_dims[idx] is a tuple of dimensions (e.g. (16,))
                            dims = term_dims[idx]
                            if len(dims) > 0:
                                self.lidar_dim = int(dims[0])
                                print(f"[MBPPO] Automatically detected lidar_dim: {self.lidar_dim} from env.")
                            else:
                                print(f"[MBPPO] Detected lidar_scan but dims are empty? {dims}")
                        else:
                            print(f"[MBPPO] 'lidar_scan' term not found in policy group. Active: {active_terms}")
        except Exception as e:
            print(f"[MBPPO] Could not auto-detect lidar dim, using default {self.lidar_dim}. Error: {e}")

        # Try to infer dt from env
        self.dt = 0.0
        try:
            base_env = getattr(self.env, "unwrapped", None)
            if base_env is not None and hasattr(base_env, "step_dt"):
                self.dt = float(base_env.step_dt)
            if self.dt == 0.0:
                # fallback to env.dt if available
                env_dt = getattr(self.env, "dt", None)
                if env_dt is not None:
                    self.dt = float(env_dt)
        except Exception:
            pass
        if self.dt > 0.0:
            print(f"[MBPPO] Detected simulation dt: {self.dt}")
        else:
            print(f"[MBPPO] Could not detect dt, using default {self.dt} (no decay)")

        # Ensure stable positive std parameterization for action distribution
        # Use log-std so that std = exp(log_std) > 0 during training
        try:
            noise_type = str(self.policy_cfg.get("noise_std_type", "scalar")).lower()
        except Exception:
            noise_type = "scalar"
        if noise_type != "log":
            self.policy_cfg["noise_std_type"] = "log"

        # actor-critic policy
        policy_class = eval(self.policy_cfg.pop("class_name"))
        self.policy: ActorCritic | ActorCriticRecurrent = policy_class(
            self.num_obs, self.num_privileged_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        # Robustly sanitize action noise parameters before every distribution update
        # to prevent invalid/negative std during sampling inside PPO.update.
        _orig_update_distribution = self.policy.update_distribution
        def _wrapped_update_distribution(self_policy, observations):
            # sanitize log_std or std in-place
            if hasattr(self_policy, "log_std") and isinstance(self_policy.log_std, torch.Tensor):
                log_std_data = self_policy.log_std.data
                invalid_mask = ~torch.isfinite(log_std_data)
                if invalid_mask.any():
                    log_std_data[invalid_mask] = 0.0
                log_std_data.clamp_(min=-5.0, max=2.0)
            if hasattr(self_policy, "std") and isinstance(self_policy.std, torch.Tensor):
                std_data = self_policy.std.data
                invalid_mask = (~torch.isfinite(std_data)) | (std_data < 0)
                if invalid_mask.any():
                    std_data[invalid_mask] = 1.0
                std_data.clamp_(min=1e-6, max=10.0)
            return _orig_update_distribution(observations)
        self.policy.update_distribution = types.MethodType(_wrapped_update_distribution, self.policy)

        # algorithm
        alg_class = eval(self.alg_cfg.pop("class_name"))
        self.alg: PPO = alg_class(self.policy, device=self.device, **self.alg_cfg)

        # training schedule
        self.num_steps_per_env: int = int(self.cfg["num_steps_per_env"])  # per real env
        self.save_interval: int = int(self.cfg["save_interval"])
        self.empirical_normalization: bool = bool(self.cfg.get("empirical_normalization", False))
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(shape=[self.num_obs], until=1_00_000_000).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(
                shape=[self.num_privileged_obs], until=1_00_000_000
            ).to(self.device)
        else:
            self.obs_normalizer = torch.nn.Identity().to(self.device)
            self.privileged_obs_normalizer = torch.nn.Identity().to(self.device)

        # model-based config
        self.num_envs_real: int = int(self.env.num_envs)
        self.num_envs_virt: int = max(0, int(mb_virtual_envs))
        self.include_virtual: bool = self.num_envs_virt > 0
        self.num_envs_total: int = int(self.num_envs_real + (self.num_envs_virt if self.include_virtual else 0))
        self.num_actions: int = int(self.env.num_actions)
        self.use_incremental_actions: bool = bool(use_incremental_actions)
        self.use_cbf: bool = bool(use_cbf)
        self.cfg["use_cbf"] = self.use_cbf  # for self.cfg.get("use_cbf", False) lookup
        # safety clamps for virtual predictions
        self.mb_virt_reward_clip: float = float(self.cfg.get("mb_virt_reward_clip", 2000.0))

        # incremental dynamics
        self.dyn_num_networks = int(num_networks)
        self.dyn_lr = float(dynamics_lr)
        self.dyn_hidden_dims = list(dynamics_hidden_dims)
        self.dyn_w_dyn = float(dyn_w_dyn)
        self.dyn_w_r = float(dyn_w_r)
        self.dyn_w_d = float(dyn_w_d)
        self.dyn_done_threshold = float(dyn_done_threshold)
        self.mb_buffer_size = int(mb_buffer_size)
        self.mb_batch_size = int(mb_batch_size)
        self.mb_update_every = int(mb_update_every)
        self.mb_update_steps = int(mb_update_steps)
        self.mb_virt_steps_per_iter = int(mb_virt_steps_per_iter)
        self.mb_warmup_iters = int(mb_warmup_iters)
        self.mb_init_from_buffer = bool(mb_init_from_buffer)
        # stability
        self.use_stability = bool(use_stability_reward)
        self.Q_scale = float(Q_scale)
        self.R_scale = float(R_scale)
        self.stability_coef = float(stability_coef)

        # dyn models and buffer
        self._dyn_models: List[IncrementalDynamicsModel] = []
        self._dyn_optims: List[torch.optim.Optimizer] = []
        self._replay: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # virtual state cache
        self._virt_states: Optional[torch.Tensor] = None
        self._virt_steps_used_in_iter: int = 0
        # action accumulators for incremental action mode
        self._real_accum_actions: Optional[torch.Tensor] = None
        self._virt_accum_actions: Optional[torch.Tensor] = None
        # previous actions for computing delta when use_incremental_actions=False
        self._prev_real_actions: Optional[torch.Tensor] = None
        self._prev_virt_actions: Optional[torch.Tensor] = None

        # MB metrics / diagnostics
        self._last_dyn_loss: float = float("nan")
        self._dyn_update_steps_iter: int = 0
        self._virt_done_count_iter: int = 0
        self._virt_step_count_iter: int = 0

        # CBF metrics / diagnostics
        self._cbf_trigger_count: int = 0           # CBF触发次数（有环境违反约束）
        self._cbf_total_violations: int = 0        # 总违规环境数
        self._cbf_total_correction: float = 0.0    # 累计动作修正量
        self._cbf_max_barrier: float = 0.0         # 最大barrier值
        self._cbf_call_count: int = 0              # CBF QP求解调用次数

        # normalization of returns
        self.alg.init_storage(
            "rl",
            self.num_envs_total,
            self.num_steps_per_env,
            [self.num_obs],
            [self.num_privileged_obs],
            [self.env.num_actions],
        )

        # debug prints control
        self._dbg_shapes = str(os.environ.get("MBPPO_DEBUG_SHAPES", "0")).strip() == "1"
        _dbg_every = str(os.environ.get("MBPPO_DEBUG_SHAPES_EVERY", "200")).strip()
        self._dbg_every = int(_dbg_every) if _dbg_every.isdigit() else 200

    # ----- internals -----
    # Lidar 归一化常量（与环境 clip=(0,5) 对应）
    LIDAR_MAX_RANGE = 5.0
    
    # CBF 安全阈值（统一定义）
    # 0.2:  最近障碍物 < 4.0m 时介入 → 最保守（最初设计）
    # 1.5:  最近障碍物 < 2.0m 时介入
    # 2.0:  最近障碍物 < 1.5m 时介入
    # 3.0:  最近障碍物 < 1.0m 时介入
    # 5.7:  最近障碍物 < 0.5m 时介入 → 激进
    CBF_SAFETY_THRESHOLD = 0.2
    
    def _normalize_lidar_in_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """对 obs 中的 lidar 部分进行归一化（原地修改后返回副本）。
        
        归一化公式：lidar_normalized = lidar / LIDAR_MAX_RANGE
        这样 lidar 从 [0, 5] 米变为 [0, 1]
        """
        obs_norm = obs.clone()
        start_idx = -self.lidar_dim
        obs_norm[:, start_idx:] = obs_norm[:, start_idx:] / self.LIDAR_MAX_RANGE
        return obs_norm
    
    def _extract_lidar(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract Lidar observations from the full observation tensor.
        
        The Lidar data is expected to be at the end of the observation vector.
        Based on ObservationsCfg in velocity_env_cfg.py:
        - UGV & UAV: ..., lidar_scan
        
        The dimension `self.lidar_dim` is auto-detected from the environment.
        Note: 如果输入是归一化后的 obs，提取的 lidar 也是归一化的 [0, 1]
        """
        start_idx = -self.lidar_dim
        return obs[:, start_idx:]

    def _compute_lidar_barrier_cost(self, obs_lidar: torch.Tensor, epsilon: float = 0.05, clip_max: float = 20.0) -> torch.Tensor:
        """
        Compute Barrier Function cost based on Lidar data (Inverse Barrier).
        B(x) = mean(1/(x + epsilon) - 1/(1 + epsilon))
        
        Note: obs_lidar 是原始 lidar 数据 [0, 5] 米，内部进行归一化
        
        使用 mean 模式：所有射线的平均危险度（最初设计）
        """
        # 内部归一化：[0, 5] 米 -> [0, 1]
        dist = torch.clamp(obs_lidar / self.LIDAR_MAX_RANGE, min=1e-6, max=1.0)
        
        # Inverse potential: f(x) = 1/(x + eps)
        barrier_raw = 1.0 / (dist + epsilon)
        
        # Zero offset: f(1) = 1/(1 + eps)
        offset = 1.0 / (1.0 + epsilon)
        barrier_cost = barrier_raw - offset
        
        # Mean over all rays: 所有射线的平均危险度（最初设计）
        total_cost = torch.mean(barrier_cost, dim=-1)
        
        # Clip to prevent gradient explosion
        total_cost = torch.clamp(total_cost, max=clip_max)
        
        return total_cost

    def _solve_cbf_qp(self, obs: torch.Tensor, u_nom: torch.Tensor) -> torch.Tensor:
        """
        Solves the CBF-QP optimization problem via Sequential Quadratic Programming (SQP).
        
        This implements the standard CBF-QP formulation:
            minimize  (1/2) * || u - u_nom ||^2
            subject to B(x_{k+1}) <= h
            
        Since the dynamics x_{k+1} = f(x, u) are non-linear (Neural Network), we linearize 
        the constraint around the current action candidate and solve the resulting QP analytically.
        
        Analytic Solution for Single-Constraint QP:
            u* = u - lambda * g
            lambda = max(0, (B - h) / ||g||^2)
            g = grad_u B(x_{k+1})
            
        We iterate this process (SQP) to handle the non-linearity of the dynamics model.
        """
        # Hyperparameters
        SQP_ITER = 3            # Number of linearization steps
        
        # 使用类常量 self.CBF_SAFETY_THRESHOLD（定义在类顶部）
        SAFETY_THRESHOLD = self.CBF_SAFETY_THRESHOLD  
        
        # Enable gradient computation even if called from no_grad context
        self._cbf_call_count += 1
        
        with torch.enable_grad():
            # Start with nominal action
            u = u_nom.detach().clone()
            u_initial = u.clone()  # 保存初始动作用于计算修正量
            u.requires_grad = True
            
            dyn_model = self._dyn_models[0]
            dyn_model.eval() # We need gradients wrt input, but not weight updates
            
            triggered = False  # 本次调用是否触发
            
            for sqp_iter in range(SQP_ITER):
                # 1. Forward pass: Predict next state x_{k+1}
                pred_next_state, _, _ = dyn_model(obs, u)
                
                # 2. Compute Barrier Value B(x_{k+1})
                lidar_obs = self._extract_lidar(pred_next_state)
                barrier_val = self._compute_lidar_barrier_cost(lidar_obs) # [num_envs]
                
                # 3. Check Constraint Violation: B(x) - h > 0
                # Note: We calculate gradients for ALL envs, even those not violating, 
                # but lambda will be 0 for safe ones.
                violation = barrier_val - SAFETY_THRESHOLD
                num_violations = (violation > 0).sum().item()
                
                # 记录统计（仅在第一次SQP迭代时）
                if sqp_iter == 0:
                    self._cbf_max_barrier = max(self._cbf_max_barrier, barrier_val.max().item())
                    if num_violations > 0:
                        triggered = True
                        self._cbf_total_violations += num_violations
                
                # Logging violation statistics for debugging
                if self._dbg_shapes and sqp_iter == 0:
                    if num_violations > 0:
                        print(f"[CBF] SQP iter {sqp_iter}: {num_violations} envs violating. Max barrier: {barrier_val.max().item():.2f}")

                # If maximum violation is negligible, we are safe.
                if violation.max() <= 1e-4:
                    break
                    
                # 4. Compute Jacobian J = grad_u B(x_{k+1})
                # We use the trick that grad(sum(B)) wrt u (batch) gives row-wise gradients
                # because samples are independent.
                sum_barrier = barrier_val.sum()
                
                # Clear previous gradients
                if u.grad is not None:
                    u.grad.zero_()
                    
                # Compute grad
                grad_u = torch.autograd.grad(sum_barrier, u, retain_graph=False)[0] # [num_envs, action_dim]
                
                # 5. Analytic QP Update (Projected Gradient)
                with torch.no_grad():
                    # Compute step size (Lagrange multiplier)
                    # lambda = (B - h) / ||grad||^2
                    grad_norm_sq = torch.sum(grad_u**2, dim=-1)
                    grad_norm_sq = torch.clamp(grad_norm_sq, min=1e-6) # Avoid div/0
                    
                    lam = violation / grad_norm_sq
                    lam = torch.relu(lam) # Constraint is inequality: only push if B > h
                    
                    # Update action
                    # u_new = u - lam * grad
                    correction = -lam.unsqueeze(-1) * grad_u
                    u += correction
                    
                    # Re-enable grad for next iteration
                    u.requires_grad = True
            
            # 统计本次调用的修正量
            if triggered:
                self._cbf_trigger_count += 1
                correction_norm = torch.norm(u.detach() - u_initial, dim=-1).mean().item()
                self._cbf_total_correction += correction_norm
            
            return u.detach()

    def _lazy_init_dyn(self, obs_dim: int, act_dim: int):
        if self._dyn_models:
            return
        # Ensure parameters are created outside inference mode
        with torch.inference_mode(False):
            for _ in range(self.dyn_num_networks):
                m = IncrementalDynamicsModel(obs_dim, act_dim, self.dyn_hidden_dims, dt=self.dt).to(self.device)
                m.train()
                opt = torch.optim.Adam(m.parameters(), lr=self.dyn_lr)
                self._dyn_models.append(m)
                self._dyn_optims.append(opt)

    def _append_replay(self, s: torch.Tensor, a: torch.Tensor, r: torch.Tensor, ns: torch.Tensor, d: torch.Tensor):
        # store real transitions only; detach to avoid graphs
        with torch.no_grad():
            for i in range(s.shape[0]):
                self._replay.append((s[i].clone(), a[i].clone(), r[i].clone(), ns[i].clone(), d[i].clone()))
            if len(self._replay) > self.mb_buffer_size:
                self._replay = self._replay[-self.mb_buffer_size :]

    def _maybe_update_dyn(self):
        if not self._dyn_models or not self._replay or len(self._replay) < max(10, self.mb_batch_size):
            return
        # Ensure autograd allowed even if called within inference_mode
        with torch.inference_mode(False):
            with torch.enable_grad():
                for m in self._dyn_models:
                    m.train()
                total_loss_sum = 0.0
                total_loss_count = 0
                for _ in range(self.mb_update_steps):
                    idx = torch.randint(low=0, high=len(self._replay), size=(self.mb_batch_size,))
                    batch = [self._replay[int(i)] for i in idx]
                    s = torch.stack([b[0] for b in batch]).to(self.device)
                    a = torch.stack([b[1] for b in batch]).to(self.device)
                    r = torch.stack([b[2] for b in batch]).to(self.device)
                    ns = torch.stack([b[3] for b in batch]).to(self.device)
                    d = torch.stack([b[4] for b in batch]).to(self.device)
                    for opt in self._dyn_optims:
                        opt.zero_grad(set_to_none=True)
                    losses = []
                    for m in self._dyn_models:
                        loss = incremental_dynamics_loss(
                            m, s, a, r, ns, d, w_dyn=self.dyn_w_dyn, w_r=self.dyn_w_r, w_d=self.dyn_w_d
                        )
                        losses.append(loss)
                    # accumulate mean loss for logging
                    if losses:
                        mean_loss_step = sum([l.detach().item() for l in losses]) / max(1, len(losses))
                        total_loss_sum += float(mean_loss_step)
                        total_loss_count += 1
                    for loss in losses:
                        loss.backward()
                    for opt in self._dyn_optims:
                        opt.step()
                    # count a dynamics update step
                    self._dyn_update_steps_iter += 1
                if total_loss_count > 0:
                    self._last_dyn_loss = total_loss_sum / float(total_loss_count)

    def _init_virtual_states(self, real_obs: torch.Tensor):
        # real_obs: (N_real, obs_dim) - 原始观测
        if self._virt_states is None:
            if self.mb_init_from_buffer and len(self._replay) > 0:
                # sample states from buffer（原始数据）
                idx = torch.randint(low=0, high=len(self._replay), size=(self.num_envs_virt,))
                states = torch.stack([self._replay[int(i)][0] for i in idx]).to(self.device)
                self._virt_states = states
            else:
                # 从 real_obs 初始化（原始数据）
                take = min(self.num_envs_virt, real_obs.shape[0])
                pad = self.num_envs_virt - take
                if pad > 0:
                    pad_states = real_obs[:1].repeat(pad, 1)
                    self._virt_states = torch.cat([real_obs[:take], pad_states], dim=0)
                else:
                    self._virt_states = real_obs[:take].clone()
        self._virt_steps_used_in_iter = 0

    # ----- training -----
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        # initialize writer
        if self.log_dir is not None and self.writer is None and not self.disable_logs:
            if self.logger_type == "neptune":
                from rsl_rl.utils.neptune_utils import NeptuneSummaryWriter

                self.writer = NeptuneSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=self.cfg)
                # best-effort: environment config may not be easily serializable
                try:
                    self.writer.log_config(getattr(self.env, "cfg", {}), self.cfg, self.alg_cfg, self.policy_cfg)
                except Exception:
                    pass
            elif self.logger_type == "wandb":
                from rsl_rl.utils.wandb_utils import WandbSummaryWriter

                # 构建 MB 参数字典，以便记录到 wandb config
                mb_cfg = {
                    "num_networks": self.dyn_num_networks,
                    "dynamics_lr": self.dyn_lr,
                    "dynamics_hidden_dims": self.dyn_hidden_dims,
                    "dyn_w_dyn": self.dyn_w_dyn,
                    "dyn_w_r": self.dyn_w_r,
                    "dyn_w_d": self.dyn_w_d,
                    "dyn_done_threshold": self.dyn_done_threshold,
                    "mb_buffer_size": self.mb_buffer_size,
                    "mb_batch_size": self.mb_batch_size,
                    "mb_update_every": self.mb_update_every,
                    "mb_update_steps": self.mb_update_steps,
                    "mb_virtual_envs": self.num_envs_virt,
                    "mb_virt_steps_per_iter": self.mb_virt_steps_per_iter,
                    "mb_warmup_iters": self.mb_warmup_iters,
                    "mb_init_from_buffer": self.mb_init_from_buffer,
                    "use_stability_reward": self.use_stability,
                    "Q_scale": self.Q_scale,
                    "R_scale": self.R_scale,
                    "stability_coef": self.stability_coef,
                    "use_incremental_actions": self.use_incremental_actions,
                    "use_cbf": self.use_cbf,
                }
                # 合并到 self.cfg 的副本中
                cfg_with_mb = {**self.cfg, "model_based": mb_cfg}
                self.writer = WandbSummaryWriter(log_dir=self.log_dir, flush_secs=10, cfg=cfg_with_mb)
                try:
                    self.writer.log_config(getattr(self.env, "cfg", {}), cfg_with_mb, self.alg_cfg, self.policy_cfg)
                except Exception:
                    pass
            elif self.logger_type == "tensorboard":
                from torch.utils.tensorboard.writer import SummaryWriter

                self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
            else:
                raise ValueError("Logger type not found. Please choose 'neptune', 'wandb' or 'tensorboard'.")
        # normalize obs buffers
        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        if self.privileged_obs_type is not None:
            priv = extras["observations"][self.privileged_obs_type].to(self.device)
        else:
            priv = obs

        # init virtual states
        if self.include_virtual:
            self._init_virtual_states(obs)
        # init accumulators for incremental actions
        if self.use_incremental_actions:
            self._real_accum_actions = torch.zeros((self.num_envs_real, self.num_actions), dtype=torch.float32, device=self.device)
            if self.include_virtual:
                self._virt_accum_actions = torch.zeros((self.num_envs_virt, self.num_actions), dtype=torch.float32, device=self.device)

        # reset per-iteration MB counters
        self._virt_steps_used_in_iter = 0
        self._virt_done_count_iter = 0
        self._virt_step_count_iter = 0
        self._dyn_update_steps_iter = 0
        # track the first step index from which virtual budget is exhausted (for PPO masking)
        _virt_frozen_from_step: int = -1

        # book-keeping
        ep_infos: List[Dict[str, Any]] = []
        rewbuffer: deque = deque(maxlen=100)
        lenbuffer: deque = deque(maxlen=100)
        cur_reward_sum = torch.zeros((self.num_envs_total,), dtype=torch.float32, device=self.device)
        cur_episode_length = torch.zeros((self.num_envs_total,), dtype=torch.float32, device=self.device)

        start_iter = 0
        tot_iter = num_learning_iterations
        for it in range(start_iter, tot_iter):
            t0 = time.time()
            # reset per-iteration MB counters so ratios reflect current iteration only
            self._virt_steps_used_in_iter = 0
            self._virt_done_count_iter = 0
            self._virt_step_count_iter = 0
            self._dyn_update_steps_iter = 0
            # reset per-iteration CBF counters
            self._cbf_trigger_count = 0
            self._cbf_total_violations = 0
            self._cbf_total_correction = 0.0
            self._cbf_max_barrier = 0.0
            self._cbf_call_count = 0
            # rollout
            # Important: use no_grad (not inference_mode) so that tensors stored in
            # rollout can still be used later in autograd during PPO.update.
            with torch.no_grad():
                for step in range(self.num_steps_per_env):
                    warmup_active = (it < self.mb_warmup_iters)
                    # enforce per-iteration virtual step budget if configured
                    if self.mb_virt_steps_per_iter > 0 and self.num_envs_virt > 0:
                        _virt_budget = int(self.mb_virt_steps_per_iter) * int(self.num_envs_virt)
                        _virt_budget_exhausted = (self._virt_steps_used_in_iter >= _virt_budget)
                    else:
                        _virt_budget_exhausted = False
                    include_virtual_step = (self.include_virtual and (not _virt_budget_exhausted))
                    if _virt_budget_exhausted and _virt_frozen_from_step < 0:
                        _virt_frozen_from_step = int(step)
                    # build combined obs
                    if self.include_virtual:
                        virt_states = self._virt_states if self._virt_states is not None else obs[:0]
                        obs_comb = torch.cat([obs, virt_states], dim=0)
                        if self.privileged_obs_type is not None:
                            priv_comb = torch.cat([priv, virt_states], dim=0)
                        else:
                            priv_comb = obs_comb
                    else:
                        obs_comb = obs
                        priv_comb = priv

                    # sample actions for combined obs (use normalized observations for stability)
                    obs_curr_nrm = self.obs_normalizer(obs_comb)
                    if self.privileged_obs_type is not None:
                        priv_curr_nrm = self.privileged_obs_normalizer(priv_comb)
                    else:
                        priv_curr_nrm = obs_curr_nrm
                    actions = self.alg.act(obs_curr_nrm, priv_curr_nrm)
                    actions_tensor: torch.Tensor = actions  # type: ignore[assignment]
                    # split for real/virtual
                    if self.use_incremental_actions:
                        # actions represent delta
                        if self.include_virtual:
                            d_real, d_virt = torch.split(actions_tensor, [self.num_envs_real, self.num_envs_virt], dim=0)
                        else:
                            d_real = actions_tensor
                            d_virt = torch.empty((0, self.num_actions), device=self.device)
                        # convert to absolute for env execution
                        if self._real_accum_actions is None:
                            self._real_accum_actions = torch.zeros((self.num_envs_real, self.num_actions), dtype=torch.float32, device=self.device)
                        a_real = self._real_accum_actions + d_real
                        self._real_accum_actions = a_real.detach()
                        a_virt_input = d_virt  # model expects same representation as training (delta)
                    else:
                        # absolute action mode
                        if self.include_virtual:
                            a_real, a_virt_abs = torch.split(actions_tensor, [self.num_envs_real, self.num_envs_virt], dim=0)
                            # compute delta for dynamics model: delta = a_t - a_{t-1}
                            if self._prev_virt_actions is None:
                                self._prev_virt_actions = torch.zeros_like(a_virt_abs)
                            a_virt_input = a_virt_abs - self._prev_virt_actions  # delta for model
                            self._prev_virt_actions = a_virt_abs.detach().clone()
                        else:
                            a_real = actions_tensor
                            a_virt_input = torch.empty((0, self.num_actions), device=self.device)

                    # step real env
                    # ----------------------------------------------------------------------
                    # CBF Safety Filter using Learned Dynamics
                    # ----------------------------------------------------------------------
                    # Apply CBF correction if dynamics models are available
                    # 性能优化：只在模型预热完成后启用CBF，避免早期无效计算
                    cbf_enabled = (
                        self._dyn_models 
                        and self.cfg.get("use_cbf", False)
                        and (not warmup_active)  # 预热期间跳过CBF
                    )
                    if cbf_enabled:
                        # 每次都调用 CBF（最初设计，无预检查）
                        obs_real_raw = obs[:self.num_envs_real]
                        if self.use_incremental_actions:
                            # MB: In incremental mode, CBF optimizes delta (d_real)
                            d_safe = self._solve_cbf_qp(obs_real_raw, d_real)
                            # Update accumulator: new_accum = (current_accum - d_nom) + d_safe
                            if self._real_accum_actions is not None:
                                self._real_accum_actions = self._real_accum_actions - d_real + d_safe
                                a_real = self._real_accum_actions.detach()
                        else:
                            a_real = self._solve_cbf_qp(obs_real_raw, a_real)
                            if self.use_incremental_actions:
                                self._real_accum_actions = a_real.detach()
                    # ----------------------------------------------------------------------

                    obs_real, rew_real, dones_real, extras_real = self.env.step(a_real.to(self.env.device))
                    obs_real = obs_real.to(self.device)
                    rew_real = rew_real.to(self.device).view(-1)  # 1D
                    dones_real = dones_real.to(self.device).view(-1)  # 1D

                    # update replay with last real transition (approximate: we need last obs, actions)
                    # Here we use current obs for state and obs_real for next_state
                    # IMPORTANT: Always store delta actions for dynamics model consistency
                    # IMPORTANT: 存储原始 obs，CBF 内部会做归一化
                    self._lazy_init_dyn(self.num_obs, self.env.num_actions)
                    with torch.no_grad():
                        # 存储原始 obs（不归一化）
                        s_real = obs[:self.num_envs_real].detach()
                        if self.use_incremental_actions:
                            # policy outputs delta directly
                            a_store = d_real.detach()
                        else:
                            # compute delta from absolute actions: delta = a_t - a_{t-1}
                            if self._prev_real_actions is None:
                                self._prev_real_actions = torch.zeros_like(a_real)
                            a_store = (a_real - self._prev_real_actions).detach()
                            self._prev_real_actions = a_real.detach().clone()
                        r_store = rew_real.detach()
                        ns_real = obs_real.detach()
                        done_real_f = (dones_real > 0).float()
                    self._append_replay(s_real, a_store, r_store, ns_real, done_real_f)

                    # build virtual next via dynamics (only after sufficient real data collected)
                    use_dyn = (
                        self.include_virtual and (not _virt_budget_exhausted)
                        and a_virt_input.shape[0] > 0
                        and self._dyn_models
                        and (len(self._replay) >= max(10, self.mb_batch_size))
                        and (self._dyn_update_steps_iter > 0)
                        and (not warmup_active)
                    )
                    if use_dyn:
                        # for rollouts, always feed delta actions (model is trained on delta)
                        m_input_actions = a_virt_input
                        vs_in = self._virt_states if self._virt_states is not None else obs[:0]
                        preds = [m(vs_in, m_input_actions) for m in self._dyn_models]
                        ns_stack = torch.stack([p[0] for p in preds], dim=0)
                        r_stack = torch.stack([p[1] for p in preds], dim=0)
                        d_stack = torch.stack([p[2] for p in preds], dim=0)
                        ns_virt = torch.mean(ns_stack, dim=0)
                        r_virt = torch.mean(r_stack, dim=0).view(-1)  # 1D
                        dprob_virt = torch.mean(d_stack, dim=0)
                        # sanitize predictions to avoid NaNs/Infs and reward explosions
                        ns_virt = torch.nan_to_num(ns_virt, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                        r_virt = torch.nan_to_num(r_virt, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                        dprob_virt = torch.nan_to_num(dprob_virt, nan=0.0, posinf=1.0, neginf=0.0)
                        r_virt = torch.clamp(r_virt, min=-self.mb_virt_reward_clip, max=self.mb_virt_reward_clip)
                        dprob_virt = torch.clamp(dprob_virt, min=1e-6, max=1.0 - 1e-6)
                        d_virt = (dprob_virt > self.dyn_done_threshold).to(dtype=torch.long).view(-1)  # 1D
                    else:
                        # fallback when not using dynamics
                        if self.include_virtual:
                            if _virt_budget_exhausted:
                                # freeze virtual states to keep dimension without consuming budget
                                if self._virt_states is not None and self._virt_states.numel() > 0:
                                    ns_virt = self._virt_states
                                else:
                                    take = min(self.num_envs_virt, obs_real.shape[0])
                                    pad = self.num_envs_virt - take
                                    ns_virt = obs_real[:take]
                                    if pad > 0:
                                        ns_virt = torch.cat([ns_virt, ns_virt[:1].repeat(pad, 1)], dim=0)
                                r_virt = torch.zeros((self.num_envs_virt,), device=self.device)
                                d_virt = torch.zeros((self.num_envs_virt,), dtype=torch.long, device=self.device)
                            else:
                                # default fallback: mirror slices of real
                                take = min(self.num_envs_virt, obs_real.shape[0])
                                pad = self.num_envs_virt - take
                                ns_virt = obs_real[:take]
                                if pad > 0:
                                    ns_virt = torch.cat([ns_virt, ns_virt[:1].repeat(pad, 1)], dim=0)
                                r_virt = torch.zeros((self.num_envs_virt,), device=self.device)
                                d_virt = torch.zeros((self.num_envs_virt,), dtype=torch.long, device=self.device)
                        else:
                            ns_virt = torch.empty((0, self.num_obs), device=self.device)
                            r_virt = torch.empty((0,), device=self.device)
                            d_virt = torch.empty((0,), dtype=torch.long, device=self.device)

                    # update MB counters
                    if include_virtual_step and d_virt is not None and d_virt.numel() > 0:
                        self._virt_steps_used_in_iter += int(d_virt.shape[0])
                        self._virt_step_count_iter += int(d_virt.shape[0])
                        self._virt_done_count_iter += int((d_virt > 0).sum().item())

                    # stability reward shaping (optional) for virtual branch
                    if self.use_stability and include_virtual_step and ns_virt.numel() > 0 and (not warmup_active):
                        # simple Lyapunov-like shaping using identity P scaled by Q_scale
                        P = torch.eye(self.num_obs, dtype=torch.float32, device=self.device) * float(self.Q_scale)
                        def V(x: torch.Tensor) -> torch.Tensor:
                            return (x @ P * x).sum(dim=-1, keepdim=True)
                        virt_states = self._virt_states if self._virt_states is not None else ns_virt
                        r_virt = r_virt + (-float(self.stability_coef) * (V(ns_virt).view(-1) - V(virt_states).view(-1)))
                    # stability reward shaping for real branch
                    if self.use_stability:
                        P = torch.eye(self.num_obs, dtype=torch.float32, device=self.device) * float(self.Q_scale)
                        def Vr(x: torch.Tensor) -> torch.Tensor:
                            return (x @ P * x).sum(dim=-1, keepdim=True)
                        rew_real = rew_real + (-float(self.stability_coef) * (Vr(obs_real).view(-1) - Vr(obs).view(-1)))

                    # concat combined outputs for PPO
                    if self.include_virtual:
                        obs_comb_next = torch.cat([obs_real, ns_virt], dim=0)
                        # rewards/dones are 1D
                        rew_comb = torch.cat([rew_real, r_virt], dim=0)
                        dones_comb = torch.cat([dones_real, d_virt], dim=0)
                        extras_comb: Dict[str, Any] = {"observations": {"policy": obs_comb_next}}
                        if self.privileged_obs_type is not None:
                            extras_comb["observations"][self.privileged_obs_type] = obs_comb_next
                        # time_outs must be 1D; PPO will unsqueeze(1) internally
                        if "time_outs" in extras_real:
                            t_out_1d = extras_real["time_outs"].to(self.device).view(-1)
                            virt_1d = torch.zeros((self.num_envs_virt,), dtype=t_out_1d.dtype, device=self.device)
                            extras_comb["time_outs"] = torch.cat([t_out_1d, virt_1d], dim=0)
                    else:
                        obs_comb_next = obs_real
                        rew_comb = rew_real
                        dones_comb = dones_real
                        extras_comb = {"observations": extras_real["observations"]}
                        if "time_outs" in extras_real:
                            extras_comb["time_outs"] = extras_real["time_outs"].to(self.device).view(-1)

                    # debug shapes
                    if self._dbg_shapes and (it * self.num_steps_per_env + step + 1) % max(1, self._dbg_every) == 0:
                        print(
                            f"[MBPPO DEBUG] step {it}:{step} obs{tuple(obs_comb_next.shape)} rew{tuple(rew_comb.shape)} dones{tuple(dones_comb.shape)}",
                            flush=True,
                        )

                    # normalize and process
                    obs_nrm = self.obs_normalizer(obs_comb_next)
                    if self.privileged_obs_type is not None:
                        priv_nrm = self.privileged_obs_normalizer(obs_comb_next)
                    else:
                        priv_nrm = obs_nrm
                    self.alg.process_env_step(rew_comb, dones_comb, {**extras_comb})

                    # bookkeeping
                    cur_reward_sum += rew_comb
                    cur_episode_length += 1
                    # collect episode/log infos from real env branch for logging (align with OnPolicyRunner)
                    if self.log_dir is not None:
                        if "episode" in extras_real:
                            ep_infos.append(extras_real["episode"])
                        elif "log" in extras_real:
                            ep_infos.append(extras_real["log"])
                    new_ids = (dones_comb > 0).nonzero(as_tuple=False).view(-1)
                    if new_ids.numel() > 0:
                        rewbuffer.extend(cur_reward_sum[new_ids].detach().cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids].detach().cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                    # reset/fold virtual states on done and reset accumulators if incremental
                    if include_virtual_step and ns_virt is not None:
                        # for done virtual envs, reset to real slice (原始数据)
                        done_mask = (d_virt > 0).view(-1)
                        if done_mask.any():
                            # 使用原始 obs reset
                            reset_src = obs_real[: self.num_envs_virt]
                            ns_virt = torch.where(done_mask.unsqueeze(-1), reset_src, ns_virt)
                            # reset action trackers for done virtual envs
                            if self.use_incremental_actions and self._virt_accum_actions is not None:
                                self._virt_accum_actions[done_mask] = 0.0
                            if not self.use_incremental_actions and self._prev_virt_actions is not None:
                                self._prev_virt_actions[done_mask] = 0.0
                        self._virt_states = ns_virt.detach()

                    # update real obs/priv
                    obs = obs_real.detach()
                    if self.privileged_obs_type is not None:
                        priv = obs_real.detach()

                    # reset real accumulators/prev_actions where done
                    done_mask_real = (dones_real > 0).view(-1)
                    if done_mask_real.any():
                        if self.use_incremental_actions and self._real_accum_actions is not None:
                            self._real_accum_actions[done_mask_real] = 0.0
                        if not self.use_incremental_actions and self._prev_real_actions is not None:
                            self._prev_real_actions[done_mask_real] = 0.0

                    # dynamics update schedule on real data
                    if (it * self.num_steps_per_env + step + 1) % max(1, self.mb_update_every) == 0:
                        self._maybe_update_dyn()

            collection_time = float(time.time() - t0)
            t1 = time.time()
            # compute returns and update policy
            # Important: bootstrap with the same env dimension used in storage (real + virtual)
            if self.include_virtual:
                virt_states_bs = self._virt_states if self._virt_states is not None else obs[:0]
                obs_bootstrap = torch.cat([obs, virt_states_bs], dim=0)
            else:
                obs_bootstrap = obs
            # apply the same normalization pathway used during rollouts
            if self.privileged_obs_type is not None:
                priv_bootstrap = self.privileged_obs_normalizer(obs_bootstrap)
                self.alg.compute_returns(priv_bootstrap)
            else:
                obs_bootstrap_nrm = self.obs_normalizer(obs_bootstrap)
                self.alg.compute_returns(obs_bootstrap_nrm)
            # After computing returns/advantages, mask out frozen-virtual contributions so PPO ignores them
            if self.include_virtual and _virt_frozen_from_step >= 0:
                stg = self.alg.storage
                # slice for steps from frozen_step to end, virtual envs only
                s_from = int(_virt_frozen_from_step)
                e_from = int(self.num_envs_real)
                # zero advantages for virtual part
                stg.advantages[s_from:, e_from:, :] = 0.0
                # set returns equal to values for virtual part -> zero value loss
                stg.returns[s_from:, e_from:, :] = stg.values[s_from:, e_from:, :]
            # guard against invalid std values before update (rare but can happen early on)
            # sanitize NaN/Inf and clamp ranges to ensure std >= 0
            if hasattr(self.alg.policy, "log_std") and isinstance(self.alg.policy.log_std, torch.Tensor):
                log_std_data = self.alg.policy.log_std.data
                invalid_mask = ~torch.isfinite(log_std_data)
                if invalid_mask.any():
                    log_std_data[invalid_mask] = 0.0
                self.alg.policy.log_std.data = torch.clamp(log_std_data, min=-5.0, max=2.0)
            if hasattr(self.alg.policy, "std") and isinstance(self.alg.policy.std, torch.Tensor):
                std_data = self.alg.policy.std.data
                invalid_mask = (~torch.isfinite(std_data)) | (std_data < 0)
                if invalid_mask.any():
                    std_data[invalid_mask] = 1.0
                self.alg.policy.std.data = torch.clamp(std_data, min=1e-6, max=10.0)
            loss_dict = self.alg.update()
            learn_time = float(time.time() - t1)

            # aligned logging & printing
            self.current_learning_iteration = it
            if self.log_dir is not None and not self.disable_logs:
                # prepare locals for logger
                locs = {
                    "collection_time": collection_time,
                    "learn_time": learn_time,
                    "ep_infos": ep_infos,
                    "rewbuffer": rewbuffer,
                    "lenbuffer": lenbuffer,
                    "loss_dict": loss_dict,
                    "it": it + 1,
                    "tot_iter": tot_iter,
                    "start_iter": start_iter + 1,
                    "num_learning_iterations": num_learning_iterations,
                }
                self.log(locs)
                # save checkpoints
                if (it + 1) % max(1, self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it+1}.pt"))
            else:
                fps = int(self.num_envs_total * self.num_steps_per_env / max(1e-6, (collection_time + learn_time)))
                print(
                    f"################################################################################\n"
                    f"                      Learning iteration {it+1}/{tot_iter}                       \n\n"
                    f"                       Computation: {fps} steps/s (collection: {collection_time:.3f}s, learning {learn_time:.3f}s)\n"
                    f"             Mean action noise std: {self.alg.policy.action_std.mean().item():.2f}\n"
                    f"          Mean value_function loss: {float(loss_dict.get('value_function', float('nan'))):.4f}\n"
                    f"               Mean surrogate loss: {float(loss_dict.get('surrogate', float('nan'))):.4f}\n"
                    f"                 Mean entropy loss: {float(loss_dict.get('entropy', float('nan'))):.4f}\n"
                    f"--------------------------------------------------------------------------------\n",
                    flush=True,
                )

    # compatibility stubs with original runner API
    def add_git_repo_to_log(self, _: str):
        return None

    def load(self, path: str, load_optimizer: bool = True):
        if not isinstance(path, str) or len(path) == 0:
            raise ValueError("Checkpoint path must be a non-empty string.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        resumed_training = False
        if "model_state_dict" in ckpt:
            resumed_training = bool(self.alg.policy.load_state_dict(ckpt["model_state_dict"]))
        if load_optimizer and resumed_training and "optimizer_state_dict" in ckpt:
            self.alg.optimizer.load_state_dict(ckpt["optimizer_state_dict"]) 
        if self.empirical_normalization:
            if "obs_norm_state_dict" in ckpt:
                self.obs_normalizer.load_state_dict(ckpt["obs_norm_state_dict"])
            if "privileged_obs_norm_state_dict" in ckpt and self.privileged_obs_type is not None:
                self.privileged_obs_normalizer.load_state_dict(ckpt["privileged_obs_norm_state_dict"])
        # also load dynamics ensemble if present in checkpoint
        if "dyn_state_dicts" in ckpt:
            self._lazy_init_dyn(self.num_obs, self.env.num_actions)
            dyn_sd = ckpt["dyn_state_dicts"]
            if len(dyn_sd) != len(self._dyn_models):
                raise ValueError(
                    f"Dynamics ensemble size mismatch: ckpt={len(dyn_sd)} vs runner={len(self._dyn_models)}."
                )
            for i, m in enumerate(self._dyn_models):
                m.load_state_dict(dyn_sd[i])
            if load_optimizer and "dyn_optimizer_state_dicts" in ckpt:
                dopts = ckpt["dyn_optimizer_state_dicts"]
                if len(dopts) != len(self._dyn_optims):
                    raise ValueError(
                        f"Dynamics optimizer count mismatch: ckpt={len(dopts)} vs runner={len(self._dyn_optims)}."
                    )
                for i, opt in enumerate(self._dyn_optims):
                    opt.load_state_dict(dopts[i])
        if "iter" in ckpt:
            self.current_learning_iteration = int(ckpt["iter"])  # informational only
        return None

    def load_dynamics(self, path: str, load_optimizers: bool = False):
        if not isinstance(path, str) or len(path) == 0:
            raise ValueError("Checkpoint path must be a non-empty string.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        ckpt = torch.load(path, map_location=self.device)
        if "dyn_state_dicts" not in ckpt:
            raise KeyError("'dyn_state_dicts' not found in checkpoint. It may not contain dynamics models.")
        # ensure dynamics models are constructed
        self._lazy_init_dyn(self.num_obs, self.env.num_actions)
        dyn_sd = ckpt["dyn_state_dicts"]
        if len(dyn_sd) != len(self._dyn_models):
            raise ValueError(
                f"Dynamics ensemble size mismatch: ckpt={len(dyn_sd)} vs runner={len(self._dyn_models)}. "
                f"Please set --num_networks accordingly."
            )
        for i, m in enumerate(self._dyn_models):
            m.load_state_dict(dyn_sd[i])
        if load_optimizers:
            if "dyn_optimizer_state_dicts" in ckpt:
                dopts = ckpt["dyn_optimizer_state_dicts"]
                if len(dopts) != len(self._dyn_optims):
                    raise ValueError(
                        f"Dynamics optimizer count mismatch: ckpt={len(dopts)} vs runner={len(self._dyn_optims)}."
                    )
                for i, opt in enumerate(self._dyn_optims):
                    opt.load_state_dict(dopts[i])
        return None

    # ---- aligned helpers ----
    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self.empirical_normalization:
            saved_dict["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
            saved_dict["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        # save dynamics ensemble if present
        if self._dyn_models:
            saved_dict["dyn_state_dicts"] = [m.state_dict() for m in self._dyn_models]
            saved_dict["dyn_optimizer_state_dicts"] = [opt.state_dict() for opt in self._dyn_optims]
        torch.save(saved_dict, path)
        # 不上传模型到 wandb/neptune，只保存到本地

    def log(self, locs: Dict[str, Any], width: int = 80, pad: int = 35):
        # Compute collection size using total envs (real + virtual)
        collection_size = self.num_steps_per_env * self.num_envs_total
        # Update totals
        self.tot_timesteps += collection_size
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        # episode info to writer
        ep_string = ""
        if locs["ep_infos"] and self.writer is not None:
            ep_reward_lines: List[str] = []
            ep_term_lines: List[str] = []
            metrics_lines: List[str] = []
            other_lines: List[str] = []
            for key in locs["ep_infos"][0]:
                try:
                    import statistics
                except Exception:
                    statistics = None
                # aggregate numeric values if present
                values = []
                for ep_info in locs["ep_infos"]:
                    if key in ep_info:
                        v = ep_info[key]
                        try:
                            v = float(v if not hasattr(v, "item") else v.item())
                            values.append(v)
                        except Exception:
                            pass
                if values:
                    mean_val = sum(values) / max(1, len(values)) if statistics is None else statistics.mean(values)
                    tag = key if "/" in key else f"Episode/{key}"
                    try:
                        self.writer.add_scalar(tag, mean_val, locs["it"])  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    # terminal label mirrors original runner
                    print_label = key if "/" in key else f"Mean episode {key}"
                    line = f"""{print_label + ':':>{pad}} {mean_val:.4f}\n"""
                    # group well-known categories first
                    tag_for_group = tag  # includes Episode/ prefix if needed
                    if tag_for_group.startswith("Episode/Reward/"):
                        ep_reward_lines.append(line)
                    elif tag_for_group.startswith("Episode/Termination/"):
                        ep_term_lines.append(line)
                    elif tag_for_group.startswith("Metrics/"):
                        metrics_lines.append(line)
                    else:
                        other_lines.append(line)
            # order: reward components, termination components, metrics, then others
            ep_string = "".join(ep_reward_lines + ep_term_lines + metrics_lines + other_lines)

        mean_std = float(self.alg.policy.action_std.mean().item())
        fps = int(collection_size / max(1e-6, iteration_time))

        # write losses and perf to writer
        if self.writer is not None:
            for key, value in locs["loss_dict"].items():
                try:
                    self.writer.add_scalar(f"Loss/{key}", value, locs["it"])  # type: ignore[attr-defined]
                except Exception:
                    pass
            try:
                self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("Policy/mean_noise_std", mean_std, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("Perf/total_fps", fps, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])  # type: ignore[attr-defined]
                if len(locs["rewbuffer"]) > 0:
                    import statistics

                    self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar(
                        "Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"],  # type: ignore[attr-defined]
                    )
                    # time-based metrics (align with OnPolicyRunner)
                    if getattr(self, "logger_type", "tensorboard") != "wandb":
                        self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)  # type: ignore[attr-defined]
                        self.writer.add_scalar(
                            "Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time,  # type: ignore[attr-defined]
                        )
                # MB-specific scalars
                # virt_done_rate = (float(self._virt_done_count_iter) / float(max(1, self._virt_step_count_iter)))
                self.writer.add_scalar("MB/num_virtual_envs", self.num_envs_virt, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/replay_size", len(self._replay), locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/virtual_steps", self._virt_steps_used_in_iter, locs["it"])  # type: ignore[attr-defined]
                # ratio of virtual data among actually collected transitions this iteration
                actual_total_steps = int(self.num_steps_per_env * self.num_envs_real) + int(self._virt_steps_used_in_iter)
                virt_ratio = float(self._virt_steps_used_in_iter) / float(max(1, actual_total_steps))
                self.writer.add_scalar("MB/virtual_ratio", virt_ratio, locs["it"])  # type: ignore[attr-defined]
                # self.writer.add_scalar("MB/virt_done_rate", virt_done_rate, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/warmup_active", int(self.current_learning_iteration < self.mb_warmup_iters), locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/dynamics_updates", self._dyn_update_steps_iter, locs["it"])  # type: ignore[attr-defined]
                if self._dyn_update_steps_iter > 0:
                    self.writer.add_scalar("MB/dynamics_last_loss", self._last_dyn_loss, locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/use_incremental_actions", int(self.use_incremental_actions), locs["it"])  # type: ignore[attr-defined]
                self.writer.add_scalar("MB/use_stability_reward", int(self.use_stability), locs["it"])  # type: ignore[attr-defined]
                if self.use_stability:
                    self.writer.add_scalar("MB/Q_scale", self.Q_scale, locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar("MB/R_scale", self.R_scale, locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar("MB/stability_coef", self.stability_coef, locs["it"])  # type: ignore[attr-defined]
                # CBF-specific scalars
                if self.use_cbf:
                    self.writer.add_scalar("CBF/trigger_count", self._cbf_trigger_count, locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar("CBF/total_violations", self._cbf_total_violations, locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar("CBF/max_barrier", self._cbf_max_barrier, locs["it"])  # type: ignore[attr-defined]
                    self.writer.add_scalar("CBF/qp_calls", self._cbf_call_count, locs["it"])  # type: ignore[attr-defined]
                    if self._cbf_trigger_count > 0:
                        avg_correction = self._cbf_total_correction / self._cbf_trigger_count
                        self.writer.add_scalar("CBF/avg_correction", avg_correction, locs["it"])  # type: ignore[attr-defined]
                    trigger_rate = self._cbf_trigger_count / max(1, self._cbf_call_count)
                    self.writer.add_scalar("CBF/trigger_rate", trigger_rate, locs["it"])  # type: ignore[attr-defined]
            except Exception:
                pass

        # terminal print
        title = f" \u001b[1m Learning iteration {locs['it']}/{locs['tot_iter']} \u001b[0m "
        if len(locs["rewbuffer"]) > 0:
            import statistics

            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'Mean {key} loss:':>{pad}} {value:.4f}\n"""
            log_string += f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
            log_string += f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            # MB-specific prints
            virt_done_rate = (float(self._virt_done_count_iter) / float(max(1, self._virt_step_count_iter)))
            log_string += f"""{'MB virtual envs:':>{pad}} {self.num_envs_virt}\n"""
            log_string += f"""{'MB replay size:':>{pad}} {len(self._replay)}\n"""
            log_string += f"""{'MB virt done rate:':>{pad}} {virt_done_rate:.2f}\n"""
            if self._dyn_update_steps_iter > 0:
                log_string += f"""{'MB dyn last loss:':>{pad}} {self._last_dyn_loss:.4f}\n"""
            # CBF-specific prints
            if self.use_cbf:
                if self._cbf_call_count > 0:
                    cbf_trigger_rate = self._cbf_trigger_count / self._cbf_call_count
                    log_string += f"""{'CBF trigger rate:':>{pad}} {cbf_trigger_rate:.1%} ({self._cbf_trigger_count}/{self._cbf_call_count} triggered)\n"""
                log_string += f"""{'CBF max barrier:':>{pad}} {self._cbf_max_barrier:.2f}\n"""
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{title.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std:.2f}\n"""
            )
            for key, value in locs["loss_dict"].items():
                log_string += f"""{f'{key}:':>{pad}} {value:.4f}\n"""
            # MB-specific prints even without reward buffer
            virt_done_rate = (float(self._virt_done_count_iter) / float(max(1, self._virt_step_count_iter)))
            log_string += f"""{'MB virtual envs:':>{pad}} {self.num_envs_virt}\n"""
            log_string += f"""{'MB replay size:':>{pad}} {len(self._replay)}\n"""
            log_string += f"""{'MB virt done rate:':>{pad}} {virt_done_rate:.2f}\n"""
            if self._dyn_update_steps_iter > 0:
                log_string += f"""{'MB dyn last loss:':>{pad}} {self._last_dyn_loss:.4f}\n"""
            # CBF-specific prints
            if self.use_cbf:
                if self._cbf_call_count > 0:
                    cbf_trigger_rate = self._cbf_trigger_count / self._cbf_call_count
                    log_string += f"""{'CBF trigger rate:':>{pad}} {cbf_trigger_rate:.1%} ({self._cbf_trigger_count}/{self._cbf_call_count} triggered)\n"""
                log_string += f"""{'CBF max barrier:':>{pad}} {self._cbf_max_barrier:.2f}\n"""

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Time elapsed:':>{pad}} {time.strftime("%H:%M:%S", time.gmtime(self.tot_time))}\n"""
            f"""{'ETA:':>{pad}} {time.strftime(
                "%H:%M:%S",
                time.gmtime(
                    (self.tot_time / max(1, (locs['it'] - locs['start_iter'] + 1)))
                    * max(0, (locs['start_iter'] + locs['num_learning_iterations'] - locs['it']))
                )
            )}\n"""
        )
        print(log_string, flush=True)


