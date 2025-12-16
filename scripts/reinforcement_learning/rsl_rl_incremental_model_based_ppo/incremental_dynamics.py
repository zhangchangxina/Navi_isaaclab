import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalDynamicsModel(nn.Module):
    """d4rl_demo-style incremental dynamics with optional reward/done heads.

    Inputs: state s, delta action du
    Core: h_B = MLP(s); B_flat = head(h_B) -> reshape to [obs_dim, act_dim]
          next_state = s + B(s) @ du
    Also returns auxiliary reward and done_prob predicted from separate MLP([s, du]).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int], dt: float = 0.0):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.dt = dt
        
        # Backbone for B(s) -> only depends on state
        dims_B = [obs_dim] + list(hidden_dims)
        layers_B = []
        for i in range(len(dims_B) - 1):
            layers_B.append(nn.Linear(dims_B[i], dims_B[i + 1]))
            layers_B.append(nn.ReLU())
        self.backbone_B = nn.Sequential(*layers_B)
        self.B_head = nn.Linear(dims_B[-1], obs_dim * act_dim)

        # Backbone for Reward/Done -> depends on state + action
        dims_aux = [obs_dim + act_dim] + list(hidden_dims)
        layers_aux = []
        for i in range(len(dims_aux) - 1):
            layers_aux.append(nn.Linear(dims_aux[i], dims_aux[i + 1]))
            layers_aux.append(nn.ReLU())
        self.backbone_aux = nn.Sequential(*layers_aux)
        
        # optional heads for compatibility
        self.reward_head = nn.Linear(dims_aux[-1], 1)
        self.done_head = nn.Linear(dims_aux[-1], 1)

    def forward(self, state: torch.Tensor, delta_action: torch.Tensor):
        # 1. Dynamics Model B(s)
        h_B = self.backbone_B(state)
        B_flat = self.B_head(h_B)
        B = B_flat.view(-1, self.obs_dim, self.act_dim)
        
        du = delta_action.unsqueeze(-1)
        delta = torch.bmm(B, du).squeeze(-1)
        next_state = (1.0 - self.dt) * state + delta

        # 2. Reward/Done Model (s, a)
        x_aux = torch.cat([state, delta_action], dim=-1)
        h_aux = self.backbone_aux(x_aux)
        
        reward = self.reward_head(h_aux).squeeze(-1)
        done_prob = torch.sigmoid(self.done_head(h_aux)).squeeze(-1)
        return next_state, reward, done_prob


def incremental_dynamics_loss(
    model: IncrementalDynamicsModel,
    state: torch.Tensor,
    delta_action: torch.Tensor,
    reward: torch.Tensor,
    next_state: torch.Tensor,
    done: torch.Tensor,
    w_dyn: float = 1.0,
    w_r: float = 1.0,
    w_d: float = 0.1,
):
    # Ensure we're in the correct gradient context even if caller is in inference mode
    with torch.inference_mode(False):
        with torch.enable_grad():
            # Build fresh tensors without inference flags
            state_new = state.detach().clone().requires_grad_(True)
            delta_action_new = delta_action.detach().clone().requires_grad_(True)
            reward_new = reward.detach()
            next_state_new = next_state.detach()
            done_new = done.detach()

            # Sanitize inputs to the model to avoid producing NaNs downstream
            state_in = torch.nan_to_num(state_new, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            delta_action_in = torch.nan_to_num(delta_action_new, nan=0.0, posinf=1.0e6, neginf=-1.0e6)

            pred_next, pred_r, pred_done = model(state_in, delta_action_in)

            # Sanitize predictions and targets
            pred_next = torch.nan_to_num(pred_next, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            pred_r = torch.nan_to_num(pred_r, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            pred_done = torch.nan_to_num(pred_done, nan=0.5, posinf=1.0, neginf=0.0)
            reward_new = torch.nan_to_num(reward_new, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            next_state_new = torch.nan_to_num(next_state_new, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            done_new = torch.nan_to_num(done_new, nan=0.0, posinf=1.0, neginf=0.0)

            loss_dyn = F.mse_loss(pred_next, next_state_new)
            loss_r = F.mse_loss(pred_r, reward_new)
            # Clamp probabilities to avoid log(0) and ensure valid BCE domain
            pred_done_clamped = torch.clamp(pred_done, min=1e-6, max=1.0 - 1e-6)
            done_clamped = torch.clamp(done_new, min=0.0, max=1.0)
            loss_d = F.binary_cross_entropy(pred_done_clamped, done_clamped)
            return w_dyn * loss_dyn + w_r * loss_r + w_d * loss_d


