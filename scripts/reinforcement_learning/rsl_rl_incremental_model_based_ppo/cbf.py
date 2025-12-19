from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import json

import torch

# #region agent log
_DEBUG_LOG_PATH = "/media/nudt3090/XYQ/ZCX/navigation/Navi_IsaacLab/.cursor/debug.log"
def _log_debug(location: str, message: str, data: dict, hypothesis_id: str = ""):
    import time
    entry = {"location": location, "message": message, "data": data, "timestamp": time.time(), "hypothesisId": hypothesis_id}
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass
# #endregion


@dataclass
class CBFMetrics:
    """Per-iteration metrics for the CBF safety filter."""

    trigger_count: int = 0          # how many times any env violated and triggered correction
    total_violations: int = 0       # total number of violating envs across all calls
    total_correction: float = 0.0   # accumulated norm of action corrections
    max_barrier: float = 0.0        # maximum barrier value observed
    call_count: int = 0             # how many times the QP solver was invoked

    def reset(self) -> None:
        self.trigger_count = 0
        self.total_violations = 0
        self.total_correction = 0.0
        self.max_barrier = 0.0
        self.call_count = 0


class LidarCBF:
    """Control Barrier Function based safety filter using Lidar and learned dynamics.

    This module is independent from the PPO runner and only depends on:
    - a dynamics model with signature: dyn_model(obs, action) -> (next_state, reward, done_prob)
    - the shape of the Lidar slice in the state vector.
    """

    def __init__(
        self,
        lidar_dim: int,
        *,
        safety_threshold: float = 0.2,
        debug: bool = False,
    ) -> None:
        self.lidar_dim: int = int(lidar_dim)
        self.safety_threshold: float = float(safety_threshold)
        self.debug: bool = bool(debug)

        self.metrics: CBFMetrics = CBFMetrics()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def reset_iteration_metrics(self) -> None:
        """Reset per-iteration statistics. Should be called once at the start of each PPO iteration."""
        self.metrics.reset()

    # Properties used by the runner for logging
    @property
    def trigger_count(self) -> int:
        return self.metrics.trigger_count

    @property
    def total_violations(self) -> int:
        return self.metrics.total_violations

    @property
    def total_correction(self) -> float:
        return self.metrics.total_correction

    @property
    def max_barrier(self) -> float:
        return self.metrics.max_barrier

    @property
    def call_count(self) -> int:
        return self.metrics.call_count

    # ---------------------------------------------------------------------
    # Core CBF logic
    # ---------------------------------------------------------------------
    def _extract_lidar(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract Lidar observations from the full observation tensor.

        The Lidar data is expected to be at the end of the observation vector.
        Dimension `self.lidar_dim` must match the environment configuration.
        """
        start_idx = -self.lidar_dim
        return obs[:, start_idx:]

    def _compute_lidar_barrier_cost(
        self,
        obs_lidar: torch.Tensor,
        epsilon: float = 0.05,
        clip_max: float = 20.0,
    ) -> torch.Tensor:
        """Compute inverse barrier cost from Lidar distances.

        Assumes obs_lidar is normalized to [0, 1] range.
        Barrier: B(x) = mean(1/(x + eps) - 1/(1 + eps))
        
        阈值含义（单根射线 Cost 参考）:
        - x=1.0 -> 0.0
        - x=0.5 -> 0.87
        - x=0.2 -> 3.05
        - x=0.1 -> 5.71
        
        对于 mean 模式，threshold=0.2 相当于大约 7% 的平均势能（即整体较靠近障碍物或有部分极近）
        """
        # 归一化: [0, 5m] -> [0, 1]，然后 clamp 防止异常
        dist = torch.clamp(obs_lidar / 5.0, min=1e-6, max=1.0)
        
        # 逆势能: f(x) = 1/(x + eps)
        barrier_raw = 1.0 / (dist + epsilon)
        
        # 零点偏移: f(1) = 1/(1 + eps)，使得 x=1 时 B=0
        offset = 1.0 / (1.0 + epsilon)
        barrier_cost = barrier_raw - offset
        
        total_cost = torch.sum(barrier_cost, dim=-1)
        
        # Clip to prevent gradient explosion
        total_cost = torch.clamp(total_cost, max=clip_max)
        return total_cost

    def solve_cbf_qp(
        self,
        dyn_model: torch.nn.Module,
        obs: torch.Tensor,
        u_nom: torch.Tensor,
        *,
        sqp_iter: int = 5,
    ) -> torch.Tensor:
        """Solve the CBF-QP via Sequential Quadratic Programming.

        minimize  (1/2) * || u - u_nom ||^2
        s.t.      B(x_{k+1}) <= safety_threshold

        where x_{k+1} comes from the learned dynamics model.
        """
        self.metrics.call_count += 1

        # #region agent log
        # 只在前 5 次调用记录详细日志
        if self.metrics.call_count <= 5:
            _log_debug("cbf.py:solve_cbf_qp:entry", "CBF QP called", {
                "call_count": self.metrics.call_count,
                "obs_shape": list(obs.shape),
                "u_nom_shape": list(u_nom.shape),
                "safety_threshold": self.safety_threshold,
                "lidar_dim": self.lidar_dim,
            }, hypothesis_id="H1_cbf_called")
        # #endregion

        with torch.enable_grad():
            # start from nominal action
            u = u_nom.detach().clone()
            u_initial = u.clone()
            u.requires_grad = True

            dyn_model.eval()

            triggered = False

            for i in range(int(sqp_iter)):
                # 1. Forward pass: predict next state
                pred_next_state, _, _ = dyn_model(obs, u)

                # 2. Compute barrier value B(x_{k+1})
                lidar_obs = self._extract_lidar(pred_next_state)
                barrier_val = self._compute_lidar_barrier_cost(lidar_obs)

                # #region agent log
                # 只在前 5 次调用的第一次 SQP 迭代记录 lidar 数据
                if self.metrics.call_count <= 5 and i == 0:
                    _log_debug("cbf.py:solve_cbf_qp:lidar", "Lidar and barrier values", {
                        "lidar_min": float(lidar_obs.min().item()),
                        "lidar_max": float(lidar_obs.max().item()),
                        "lidar_mean": float(lidar_obs.mean().item()),
                        "barrier_min": float(barrier_val.min().item()),
                        "barrier_max": float(barrier_val.max().item()),
                        "barrier_mean": float(barrier_val.mean().item()),
                        "threshold": self.safety_threshold,
                    }, hypothesis_id="H2_lidar_range")
                # #endregion

                # 3. Constraint violation: B(x) - h > 0
                violation = barrier_val - self.safety_threshold
                num_violations = (violation > 0).sum().item()

                if i == 0:
                    self.metrics.max_barrier = max(
                        self.metrics.max_barrier,
                        float(barrier_val.max().item()),
                    )
                    if num_violations > 0:
                        triggered = True
                        self.metrics.total_violations += int(num_violations)

                if self.debug and i == 0 and num_violations > 0:
                    print(
                        f"[CBF] SQP iter {i}: {num_violations} envs violating. "
                        f"Max barrier: {barrier_val.max().item():.2f}",
                        flush=True,
                    )

                # If maximum violation is negligible, we are already safe.
                if violation.max() <= 1e-4:
                    break

                # 4. Jacobian J = dB/du via autograd
                sum_barrier = barrier_val.sum()

                if u.grad is not None:
                    u.grad.zero_()

                grad_u = torch.autograd.grad(
                    sum_barrier,
                    u,
                    retain_graph=False,
                )[0]

                # 5. Analytic QP update
                with torch.no_grad():
                    grad_norm_sq = torch.sum(grad_u**2, dim=-1)
                    grad_norm_sq = torch.clamp(grad_norm_sq, min=1e-6)

                    lam = violation / grad_norm_sq
                    lam = torch.relu(lam)

                    correction = -lam.unsqueeze(-1) * grad_u
                    u += correction

                    u.requires_grad = True

            if triggered:
                self.metrics.trigger_count += 1
                correction_norm = torch.norm(u.detach() - u_initial, dim=-1).mean().item()
                self.metrics.total_correction += float(correction_norm)

            return u.detach()


