# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # Return a per-env boolean tensor of False (no termination on plane)
        _asset: RigidObject = env.scene[asset_cfg.name]
        return torch.zeros((_asset.data.root_pos_w.shape[0],), dtype=torch.bool, device=_asset.data.root_pos_w.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def out_of_height_limit(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), 
    min_height: float = 1.0,
    max_height: float = 5.0
) -> torch.Tensor:
    """Terminate when the actor is outside the height limits.

    Args:
        env: The environment object.
        asset_cfg: The asset configuration.
        min_height: Minimum allowed height (m).
        max_height: Maximum allowed height (m).
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # check if the agent is out of height bounds
    z = asset.data.root_pos_w[:, 2]
    too_low = z < min_height
    too_high = z > max_height
    return torch.logical_or(too_low, too_high)


def least_lidar_depth(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    depth = torch.norm(sensor.data.ray_hits_w[..., 0:2] - sensor.data.pos_w[:, 0:2].unsqueeze(1), dim=-1)
    return torch.any(depth < threshold, dim=1)



def roll_over(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.1
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # check if the agent is out of bounds
    is_roll_over = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1) > threshold
    return is_roll_over


def reach_target(
    env: ManagerBasedRLEnv, threshold: float, command_name: str,
    velocity_threshold: float = None,  # 速度阈值 m/s, None表示不检查速度
    use_3d: bool = False  # 是否使用3D距离（UAV用True，UGV用False）
) -> torch.Tensor:
    """Terminate when the robot reaches the target within threshold distance.
    
    Args:
        env: The environment object.
        threshold: Position distance threshold in meters.
        command_name: Name of the command to track.
        velocity_threshold: Optional velocity threshold. If provided, robot must also 
                          have velocity below this value to terminate.
        use_3d: If True, use 3D distance (for UAV). If False, use 2D XY distance (for UGV).
    """
    command = env.command_manager.get_command(command_name)
    
    if use_3d:
        # Use 3D distance for UAV
        des_pos_b = command[:, :3]
    else:
        # Use 2D XY distance for UGV
        des_pos_b = command[:, :2]
    
    distance = torch.norm(des_pos_b, dim=1)
    position_reached = distance <= threshold
    
    # 如果指定了速度阈值，还需要检查速度
    if velocity_threshold is not None:
        asset: RigidObject = env.scene["robot"]
        velocity = torch.norm(asset.data.root_lin_vel_w[:, :3], dim=1)
        velocity_low = velocity <= velocity_threshold
        # 需要同时满足：位置到达 + 速度足够低
        return position_reached & velocity_low
    
    # Return True if within threshold, False otherwise
    return position_reached

