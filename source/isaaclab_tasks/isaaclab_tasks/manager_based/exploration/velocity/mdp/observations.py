# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom observation functions for exploration environments."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_height(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the height (z position) of the robot base.
    
    Returns:
        Height as a tensor of shape (num_envs, 1).
    """
    asset = env.scene["robot"]
    return asset.data.root_pos_w[:, 2:3]  # shape: (num_envs, 1)
