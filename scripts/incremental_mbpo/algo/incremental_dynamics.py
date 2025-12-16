import torch
import torch.nn as nn
import torch.nn.functional as F


class IncrementalDynamicsModel(nn.Module):
    """d4rl_demo-style incremental dynamics with optional reward/done heads.

    Inputs: state s, delta action du
    Core: h = MLP([s, du]); B_flat = head(h) -> reshape to [obs_dim, act_dim]
          next_state = s + B(s) @ du
    Also returns auxiliary reward and done_prob predicted from h (kept for compatibility).
    """

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        dims = [obs_dim + act_dim] + list(hidden_dims)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.backbone = nn.Sequential(*layers)
        self.B_head = nn.Linear(dims[-1], obs_dim * act_dim)
        # optional heads for compatibility
        self.reward_head = nn.Linear(dims[-1], 1)
        self.done_head = nn.Linear(dims[-1], 1)

    def forward(self, state: torch.Tensor, delta_action: torch.Tensor):
        x = torch.cat([state, delta_action], dim=-1)
        h = self.backbone(x)
        B_flat = self.B_head(h)
        B = B_flat.view(-1, self.obs_dim, self.act_dim)
        du = delta_action.unsqueeze(-1)
        delta = torch.bmm(B, du).squeeze(-1)
        next_state = state + delta
        reward = self.reward_head(h).squeeze(-1)
        done_prob = torch.sigmoid(self.done_head(h)).squeeze(-1)
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
    pred_next, pred_r, pred_done = model(state, delta_action)
    loss_dyn = F.mse_loss(pred_next, next_state)
    loss_r = F.mse_loss(pred_r, reward)
    loss_d = F.binary_cross_entropy(pred_done, done)
    return w_dyn * loss_dyn + w_r * loss_r + w_d * loss_d


