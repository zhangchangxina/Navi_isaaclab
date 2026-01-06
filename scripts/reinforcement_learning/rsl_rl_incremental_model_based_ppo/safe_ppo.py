# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Safe PPO algorithm with actor regularization for safety-constrained RL.

This module extends the standard PPO algorithm to handle the distribution mismatch
between the learned policy (u1) and the safety-corrected actions (u2) from a CBF filter.

Key insight from Offline RL (ORL-RC):
- The learned policy π_k (outputting u1) is analogous to the baseline policy in offline RL
- The behavior policy π_b (producing u2) is the corrected policy from the safety filter
- We use actor regularization to minimize the divergence between u1 and u2

References:
- Efficient Offline Reinforcement Learning With Relaxed Conservatism (ORL-RC)
- Conservative Q-Learning (CQL)
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic


class SafePPO(PPO):
    """PPO with actor regularization for safe RL.
    
    Extends PPO by adding a regularization term that encourages the policy
    to output actions closer to the safety-corrected actions (u2).
    
    Supports two loss types:
    - MSE: L_bc = ||u1 - u2||^2 (simple, fast)
    - KL:  L_bc = KL[π(·|s) || π_b(·|s)] (theoretically grounded, Eq 9 style)
    
    For Gaussian policies, KL divergence is:
        KL = log(σ_b/σ) + (σ² + (μ - μ_b)²) / (2σ_b²) - 0.5
    
    When assuming σ_b = σ (same variance), this simplifies to:
        KL ≈ (μ - μ_b)² / (2σ²)
    
    which is variance-weighted MSE.
    """

    def __init__(
        self,
        actor_critic: ActorCritic,
        *,
        device: str = "cpu",
        # Safe RL parameters (extracted before passing to parent)
        bc_coef: float = 0.1,
        bc_decay: float = 1.0,
        bc_min: float = 0.0,
        bc_loss_type: Literal["mse", "kl"] = "mse",
        # All other PPO parameters
        **kwargs,
    ) -> None:
        """Initialize SafePPO.
        
        Args:
            actor_critic: Actor-critic network
            device: Device to use
            bc_coef: Behavior cloning regularization coefficient (λ_bc)
            bc_decay: Decay factor for bc_coef per update (for annealing)
            bc_min: Minimum value for bc_coef after decay
            bc_loss_type: Type of BC loss - "mse" or "kl"
            **kwargs: Other PPO parameters (num_learning_epochs, clip_param, etc.)
        """
        # Pass all other kwargs to parent PPO
        super().__init__(
            actor_critic,
            device=device,
            **kwargs,
        )
        
        # Safe RL parameters
        self.bc_coef = float(bc_coef)
        self.bc_decay = float(bc_decay)
        self.bc_min = float(bc_min)
        self.bc_loss_type = str(bc_loss_type).lower()
        
        # Storage for original actions (u1) - will be set by runner
        self._original_actions: Optional[torch.Tensor] = None
        
        # Metrics for logging
        self._last_bc_loss: float = 0.0
        self._last_kl_divergence: float = 0.0
        
    def set_original_actions(self, actions: torch.Tensor) -> None:
        """Store original actions (u1) for regularization computation.
        
        This should be called by the runner before update() to provide
        the original policy outputs (before CBF correction).
        
        Args:
            actions: Original actions tensor, shape (num_steps, num_envs, action_dim)
        """
        self._original_actions = actions.detach().clone()
    
    def _compute_kl_loss(
        self, 
        mu: torch.Tensor, 
        sigma: torch.Tensor, 
        u2: torch.Tensor,
        sigma_b: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute KL divergence between current policy and behavior policy.
        
        KL[π(·|s) || π_b(·|s)] for Gaussian distributions.
        
        Args:
            mu: Mean of current policy, shape (batch, action_dim)
            sigma: Std of current policy, shape (batch, action_dim) or (action_dim,)
            u2: Safe action (treated as mean of behavior policy), shape (batch, action_dim)
            sigma_b: Std of behavior policy. If None, assumes σ_b = σ.
        
        Returns:
            KL divergence, scalar tensor
        """
        # Ensure sigma has correct shape
        if sigma.dim() == 1:
            sigma = sigma.unsqueeze(0).expand_as(mu)
        
        # If behavior policy std not provided, assume same as current policy
        if sigma_b is None:
            sigma_b = sigma
        elif sigma_b.dim() == 1:
            sigma_b = sigma_b.unsqueeze(0).expand_as(mu)
        
        # Clamp for numerical stability
        sigma = torch.clamp(sigma, min=1e-6)
        sigma_b = torch.clamp(sigma_b, min=1e-6)
        
        # KL divergence for Gaussian: 
        # KL = log(σ_b/σ) + (σ² + (μ - μ_b)²) / (2σ_b²) - 0.5
        # Here μ_b = u2 (the safe action)
        
        log_ratio = torch.log(sigma_b / sigma)
        variance_term = sigma.pow(2) / (2 * sigma_b.pow(2))
        mean_term = (mu - u2).pow(2) / (2 * sigma_b.pow(2))
        
        kl_per_dim = log_ratio + variance_term + mean_term - 0.5
        
        # Sum over action dimensions, mean over batch
        kl = kl_per_dim.sum(dim=-1).mean()
        
        return kl
        
    def update(self):  # noqa: C901
        """Perform PPO update with actor regularization.
        
        This overrides the parent PPO.update() to add BC/KL regularization loss.
        Follows the exact same structure as rsl_rl.algorithms.PPO.update().
        
        Returns:
            Dictionary of losses including 'bc_loss'
        """
        # Check if we have original actions for BC loss
        use_bc = (
            self._original_actions is not None 
            and self.bc_coef > 0
            and self._original_actions.shape == self.storage.actions.shape
        )
        
        # If not using BC, just call parent update
        if not use_bc:
            result = super().update()
            result["bc_loss"] = 0.0
            return result
        
        # Otherwise, we need to compute BC loss alongside standard PPO losses
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_bc_loss = 0.0
        mean_kl_div = 0.0
        
        # -- RND loss (same as parent PPO)
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss (same as parent PPO)
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches (same as parent PPO)
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches (same structure as parent PPO)
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation (same as parent PPO)
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], obs_type="policy"
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], obs_type="critic"
                )
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy (same as parent PPO)
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # entropy: only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL divergence for adaptive learning rate (same as parent PPO)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss (same as parent PPO)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss (same as parent PPO)
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            # Base loss (same as parent PPO)
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss (same as parent PPO)
            if self.symmetry:
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], obs_type="policy"
                    )
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], obs_type="policy"
                )

                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss (same as parent PPO)
            if self.rnd:
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # ====== BC Loss (SafePPO addition) ======
            # Compute BC loss on current policy mean vs actions_batch (u2)
            # This encourages policy to match the safe actions directly
            current_mu = self.policy.action_mean
            current_sigma = self.policy.action_std
            u2_batch = actions_batch
            
            if self.bc_loss_type == "kl":
                bc_loss = self._compute_kl_loss(
                    mu=current_mu,
                    sigma=current_sigma,
                    u2=u2_batch,
                    sigma_b=None,
                )
                kl_div_value = bc_loss.item()
            else:
                bc_loss = torch.mean((current_mu - u2_batch).pow(2))
                kl_div_value = 0.0
            
            # Add BC loss to total loss
            loss += self.bc_coef * bc_loss

            # Compute the gradients (same as parent PPO)
            self.optimizer.zero_grad()
            loss.backward()
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs (same as parent PPO)
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients (same as parent PPO)
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_bc_loss += bc_loss.item()
            mean_kl_div += kl_div_value
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Average metrics
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_bc_loss /= num_updates
        mean_kl_div /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        
        # Store for logging
        self._last_bc_loss = mean_bc_loss
        self._last_kl_divergence = mean_kl_div
        
        # Decay bc_coef
        self.bc_coef = max(self.bc_min, self.bc_coef * self.bc_decay)
        
        # Clear original actions
        self._original_actions = None
        
        # Clear storage (same as parent PPO)
        self.storage.clear()

        # Construct the loss dictionary (same as parent PPO + bc_loss)
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "bc_loss": mean_bc_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        if self.bc_loss_type == "kl":
            loss_dict["bc_kl_divergence"] = mean_kl_div

        return loss_dict
    
    @property
    def last_bc_loss(self) -> float:
        """Get the last BC loss value for logging."""
        return self._last_bc_loss
    
    @property
    def last_kl_divergence(self) -> float:
        """Get the last KL divergence value for logging (only meaningful when bc_loss_type='kl')."""
        return self._last_kl_divergence
