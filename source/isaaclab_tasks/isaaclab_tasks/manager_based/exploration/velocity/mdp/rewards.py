# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    # Support both pose commands with explicit heading (>=4 dims)
    # and position-only commands (3 dims). For the latter, infer heading
    # from the planar direction to the goal using atan2(y, x).
    if command.size(1) >= 4:
        heading_signal = command[:, 3]
    else:
        # command expected as (x, y, z?) — use x,y to derive desired yaw
        heading_signal = torch.atan2(command[:, 1], command[:, 0])
    return heading_signal.abs()


def lin_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)

def ang_vel_z_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_ang_vel_b[:, 2])


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def position_command_error_abs(env: ManagerBasedRLEnv, origin_distance: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with a bounded tanh kernel.

    Notes:
        We intentionally keep this reward bounded in [0, 1] to avoid a common failure mode where the agent
        prefers to loiter near the target (just outside the termination threshold) to collect large per-step
        rewards instead of finishing the episode.
    """
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    # normalize distance by a characteristic "origin" distance and squash
    # distance -> 0  => reward -> 1
    # distance -> inf => reward -> 0
    return 1.0 - torch.tanh(distance / origin_distance)


def position_command_error_2d(env: ManagerBasedRLEnv, origin_distance: float, command_name: str) -> torch.Tensor:
    """Reward 2D position tracking for UGV (ignoring height), bounded with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_2d = command[:, :2]  # 只取xy平面坐标
    distance_2d = torch.norm(des_pos_2d, dim=1)  # 2D欧几里得距离
    return 1.0 - torch.tanh(distance_2d / origin_distance)


def lidar_depth_min(env: ManagerBasedRLEnv, threshold: float, sensor_cfg = SceneEntityCfg("lidar_scanner")) -> torch.Tensor:
    """Penalize too close distance to obstacles."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    depth = torch.norm(sensor.data.ray_hits_w[..., 0:2] - sensor.data.pos_w[:, 0:2].unsqueeze(1), dim=-1)
    min_depth, _ = torch.min(depth, dim=-1)

    return torch.clamp(threshold - min_depth, min=0)



def reach_target_reward(env: ManagerBasedRLEnv, threshold: float, command_name: str) -> torch.Tensor:
    """Reward function for reaching target - gives large reward when within threshold."""
    from .terminations import reach_target
    is_reached = reach_target(env, threshold, command_name)
    return torch.where(is_reached, 1.0, 0.0)


def velocity_near_target_penalty(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    distance_threshold: float = 5.0,  # 5米内开始减速
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize high velocity when near target - encourages hovering/stopping.
    
    Args:
        env: The environment object.
        command_name: Name of the command to track.
        distance_threshold: Distance to target below which velocity is penalized.
        asset_cfg: Configuration for the robot asset.
        
    Returns:
        Velocity magnitude when near target, zero otherwise.
    """
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :2]
    distance = torch.norm(des_pos_b, dim=1)
    
    asset = env.scene[asset_cfg.name]
    velocity = torch.norm(asset.data.root_lin_vel_w[:, :3], dim=1)
    
    # 只在接近目标时惩罚速度
    near_target = distance < distance_threshold
    return torch.where(near_target, velocity, torch.zeros_like(velocity))
