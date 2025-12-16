# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    omni_debug_draw = None


trajectory_points = None




def vis_trajectories(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_length: int = 100,
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    pos_w = asset.data.root_pos_w[env_ids]

    global trajectory_points
    if trajectory_points is None:
           trajectory_points = pos_w.clone().unsqueeze(1).repeat(1, max_length, 1)
    else:
        trajectory_points = trajectory_points.roll(shifts=-1, dims=1)  # 沿dim=1左移1位
        trajectory_points[:, -1, :] = pos_w

    if not DEBUG_DRAW_AVAILABLE:
        return
        
    draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    draw_interface.clear_lines()
    for i in range(trajectory_points.shape[0]):
        source_pos = trajectory_points[i, :-1, :]
        target_pos = trajectory_points[i, 1:, :]

        lines_colors = [[1.0, 1.0, 0.0, 1.0]] * source_pos.shape[0]
        line_thicknesses = [5.0] * source_pos.shape[0]

        draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), lines_colors, line_thicknesses)
    



    # # velocities
    # vel_w = asset.data.root_vel_w[env_ids]
    # # sample random velocities
    # range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    # ranges = torch.tensor(range_list, device=asset.device)
    # vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # # set the velocities into the physics simulation
    # asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)






