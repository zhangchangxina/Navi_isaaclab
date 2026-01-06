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
        gamma: float = 1,
        debug: bool = False,
    ) -> None:
        self.lidar_dim: int = int(lidar_dim)
        self.gamma: float = float(gamma)
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
    def avg_correction(self) -> float:
        """Average correction norm per trigger."""
        if self.metrics.trigger_count > 0:
            return self.metrics.total_correction / self.metrics.trigger_count
        return 0.0

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

    def _compute_barrier_and_h(
        self,
        obs_lidar: torch.Tensor,
        epsilon: float = 0.05,
        clip_max: float = 10000.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute barrier cost B and metric h.

        Assumes obs_lidar is normalized to [0, 1] range.
        Barrier: B(x) = -log(x + eps) - offset
        This is numerically more stable than 1/(x+eps).
        """
        # 归一化: [0, 5m] -> [0, 1]，然后 clamp 防止异常
        dist = torch.clamp(obs_lidar / 5.0, min=1e-6, max=1.0)
        
        # Log barrier: f(x) = -log(x + eps)
        # 梯度是 -1/(x+eps)，不像 1/(x+eps)^2 那么剧烈
        barrier_raw = -torch.log(dist + epsilon)
        
        # 零点偏移: f(1) = -log(1 + eps)，使得 x=1 时 B=0
        offset = -torch.log(torch.tensor(1.0 + epsilon, device=dist.device))
        barrier_raw = barrier_raw - offset

        total_barrier = torch.sum(barrier_raw, dim=-1)
        total_h = torch.sum(dist, dim=-1)
        
        # Clip to prevent gradient explosion
        total_barrier = torch.clamp(total_barrier, max=clip_max)
        return total_barrier, total_h

    def solve_cbf_qp(
        self,
        dyn_model: torch.nn.Module,
        obs: torch.Tensor,
        u_nom: torch.Tensor,
        *,
        sqp_iter: int = 3,
    ) -> torch.Tensor:
        """Solve the CBF-QP via Sequential Quadratic Programming.

        minimize  (1/2) * || u - u_nom ||^2
        s.t.      B(x_{k+1}) - B(x_k) <= gamma * h(x_k)

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
                "gamma": self.gamma,
                "lidar_dim": self.lidar_dim,
            }, hypothesis_id="H1_cbf_called")
        # #endregion

        # Must exit inference_mode AND enable_grad for autograd to work
        # inference_mode(False) is needed because play scripts run inside inference_mode()
        with torch.inference_mode(False), torch.enable_grad():
            # Clone obs to escape inference mode (obs may have been created in inference_mode context)
            obs = obs.detach().clone()
            
            # start from nominal action
            u = u_nom.detach().clone()
            u_initial = u.clone()
            u.requires_grad = True

            dyn_model.eval()

            # Pre-compute current barrier and h
            lidar_curr = self._extract_lidar(obs)
            b_curr, h_curr = self._compute_barrier_and_h(lidar_curr)
            b_curr = b_curr.detach()
            h_curr = h_curr.detach()

            triggered = False

            for i in range(int(sqp_iter)):
                # 1. Forward pass: predict next state
                pred_next_state, _, _ = dyn_model(obs, u)

                # 2. Compute barrier value B(x_{k+1})
                lidar_next = self._extract_lidar(pred_next_state)
                b_next, _ = self._compute_barrier_and_h(lidar_next)

                # #region agent log
                # 只在前 5 次调用的第一次 SQP 迭代记录 lidar 数据
                if self.metrics.call_count <= 5 and i == 0:
                    _log_debug("cbf.py:solve_cbf_qp:lidar", "Lidar and barrier values", {
                        "lidar_min": float(lidar_next.min().item()),
                        "lidar_max": float(lidar_next.max().item()),
                        "lidar_mean": float(lidar_next.mean().item()),
                        "barrier_min": float(b_next.min().item()),
                        "barrier_max": float(b_next.max().item()),
                        "barrier_mean": float(b_next.mean().item()),
                        "gamma": self.gamma,
                    }, hypothesis_id="H2_lidar_range")
                # #endregion

                # 3. Constraint violation: B(x') - B(x) - gamma * h(x) > 0
                violation = b_next - b_curr - self.gamma * h_curr
                num_violations = (violation > 0).sum().item()

                if i == 0:
                    self.metrics.max_barrier = max(
                        self.metrics.max_barrier,
                        float(b_next.max().item()),
                    )
                    if num_violations > 0:
                        triggered = True
                        self.metrics.total_violations += int(num_violations)

                if self.debug and i == 0 and num_violations > 0:
                    print(
                        f"[CBF] SQP iter {i}: {num_violations} envs violating. "
                        f"Max violation: {violation.max().item():.2f}",
                        flush=True,
                    )

                # If maximum violation is negligible, we are already safe.
                if violation.max() <= 1e-4:
                    break

                # 4. Jacobian J = d(violation)/du via autograd
                sum_violation = violation.sum()

                if u.grad is not None:
                    u.grad.zero_()

                grad_u = torch.autograd.grad(
                    sum_violation,
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


