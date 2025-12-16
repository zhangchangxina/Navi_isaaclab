import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list[int]):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden_dims)
        self.act_dim = act_dim

    def sample(self, obs: torch.Tensor):
        mu_logstd = self.net(obs)
        mu, log_std = torch.split(mu_logstd, self.act_dim, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        log_prob = dist.log_prob(x_t).sum(dim=-1, keepdim=True)
        action = torch.tanh(x_t)
        # Tanh correction
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mu)


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden_dims)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        sa = torch.cat([obs, act], dim=-1)
        return self.q1(sa), self.q2(sa)


class SAC:
    def __init__(self, obs_dim: int, action_space, args):
        self.device = torch.device("cuda" if args.cuda else "cpu")
        hidden = args.critic_hidden_dims if hasattr(args, "critic_hidden_dims") else [256, 256]
        self.critic = QNetwork(obs_dim, action_space.shape[0], hidden).to(self.device)
        self.critic_target = QNetwork(obs_dim, action_space.shape[0], hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        pol_hidden = args.policy_hidden_dims if hasattr(args, "policy_hidden_dims") else [256, 256]
        self.policy = GaussianPolicy(obs_dim, action_space.shape[0], pol_hidden).to(self.device)

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.critic_optim = Adam(self.critic.parameters(), lr=getattr(args, "critic_lr", args.lr))
        self.policy_optim = Adam(self.policy.parameters(), lr=getattr(args, "policy_lr", args.lr))
        if self.automatic_entropy_tuning:
            self.target_entropy = -action_space.shape[0]
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=getattr(args, "alpha_lr", args.lr))

    def select_action(self, state, eval: bool = False):
        if not torch.is_tensor(state):
            state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            action, log_pi, mean_action = self.policy.sample(state)
            a = mean_action if eval else action
        return a.squeeze(0).cpu().numpy()

    def update_critic_only(self, memory, batch_size: int, updates: int):
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory
        if not torch.is_tensor(state_batch):
            state_batch = torch.as_tensor(state_batch, dtype=torch.float32)
            action_batch = torch.as_tensor(action_batch, dtype=torch.float32)
            reward_batch = torch.as_tensor(reward_batch, dtype=torch.float32)
            next_state_batch = torch.as_tensor(next_state_batch, dtype=torch.float32)
            mask_batch = torch.as_tensor(mask_batch, dtype=torch.float32)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device).unsqueeze(-1)
        next_state_batch = next_state_batch.to(self.device)
        mask_batch = mask_batch.to(self.device).unsqueeze(-1)

        with torch.no_grad():
            next_pi, next_log_pi, _ = self.policy.sample(next_state_batch)
            q1_next, q2_next = self.critic_target(next_state_batch, next_pi)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_pi
            q_target = reward_batch + mask_batch * self.gamma * min_q_next

        q1, q2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(q1, q_target)
        qf2_loss = F.mse_loss(q2, q_target)
        loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()
        return qf1_loss.item(), qf2_loss.item()

    def update_actor_only(self, memory, batch_size: int, updates: int):
        state_batch, _, _, _, _ = memory
        if not torch.is_tensor(state_batch):
            state_batch = torch.as_tensor(state_batch, dtype=torch.float32)
        state_batch = state_batch.to(self.device)
        pi, log_pi, _ = self.policy.sample(state_batch)
        q1_pi, q2_pi = self.critic(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0)

        if updates % self.target_update_interval == 0:
            for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(tp.data * (1.0 - self.tau) + p.data * self.tau)

        return policy_loss.item(), float(log_pi.mean().item()), float(self.alpha)


