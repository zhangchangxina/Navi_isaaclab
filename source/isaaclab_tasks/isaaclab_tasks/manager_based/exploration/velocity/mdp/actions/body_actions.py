# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.string as string_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm
# from isaaclab.envs.mdp.actions import JointAction
import isaaclab.utils.math as math_utils


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import actions_cfg




class BodyAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: actions_cfg.BodyActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | list[float] | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""
    _clip: torch.Tensor
    """The clip applied to the input action."""

    _action_dim: int


    def __init__(self, cfg: actions_cfg.BodyActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self._action_dim = cfg.action_dim


        # resolve the joints over which the action term is applied
        # self._joint_ids, self._joint_names = self._asset.find_joints(
        #     self.cfg.joint_names, preserve_order=self.cfg.preserve_order
        # )
        # self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging

        # parse the body index
        self._body_idx, self._body_name = self._asset.find_bodies(self.cfg.body_name)
        if len(self._body_idx) != 1:
            raise ValueError(f"Found more than one body match for the body name: {self.cfg.body_name}")

        omni.log.info(
            f"Resolved body name for the action term {self.__class__.__name__}:"
            f" {self._body_name} [{self._body_idx}]"
        )


        # Avoid indexing across all joints for efficiency
        # if self._num_joints == self._asset.num_joints and not self.cfg.preserve_order:
        #     self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, list):
            self._scale = torch.tensor(cfg.scale, device=self.device)
        elif isinstance(cfg.scale, dict):
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.scale, self._body_name)
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float and dict.")
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, dict):
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.offset, self._body_name)
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float and dict.")
        # parse clip
        if self.cfg.clip is not None:
            if isinstance(cfg.clip, dict):
                self._clip = torch.tensor([[-float("inf"), float("inf")]], device=self.device).repeat(
                    self.num_envs, self.action_dim, 1
                )
                index_list, _, value_list = string_utils.resolve_matching_names_values(self.cfg.clip, self._body_name)
                self._clip[:, index_list] = torch.tensor(value_list, device=self.device)
            else:
                raise ValueError(f"Unsupported clip type: {type(cfg.clip)}. Supported types are dict.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset
        # clip actions
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0












class UGVBodyAction(BodyAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.UGVBodyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UGVBodyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_root_state[:, self._body_idx].clone()

        self.acc_limit = cfg.acc_limit

    def apply_actions(self):
        # 获取当前线速度和角速度
        current_lin_vel = torch.norm(self._asset.data.root_lin_vel_b[:, :2], dim=-1)  # xy平面的线速度
        current_ang_vel = self._asset.data.root_ang_vel_b[:, 2]  # z轴角速度
        quat = self._asset.data.root_link_quat_w

        # 目标线速度和角速度（动作直接对应速度）
        target_lin_vel = self.processed_actions[:, 0]
        target_ang_vel = self.processed_actions[:, 1]
        
        # 使用配置中的加速度限制
        lin_acc_limit_per_step = self.cfg.lin_acc_limit_per_step
        ang_acc_limit_per_step = self.cfg.ang_acc_limit_per_step
        
        # 对线速度进行加速度限制 (基于Nav2标准)
        # 限制速度变化率，确保平滑运动
        clamped_lin_vel = torch.clamp(
            target_lin_vel, 
            min=current_lin_vel - lin_acc_limit_per_step, 
            max=current_lin_vel + lin_acc_limit_per_step
        )

        # 对角速度进行加速度限制 (基于Nav2标准)
        # 限制角速度变化率，确保平滑转向
        clamped_ang_vel = torch.clamp(
            target_ang_vel, 
            min=current_ang_vel - ang_acc_limit_per_step, 
            max=current_ang_vel + ang_acc_limit_per_step
        )

        # 计算线速度方向（基于当前xy平面速度方向或默认方向）
        current_lin_vel_direction = torch.zeros_like(self._asset.data.root_lin_vel_b[:, :2])
        non_zero_mask = current_lin_vel > 1e-6
        current_lin_vel_direction[non_zero_mask] = self._asset.data.root_lin_vel_b[non_zero_mask, :2] / current_lin_vel[non_zero_mask].unsqueeze(-1)
        
        # 如果没有当前线速度，使用默认的前进方向
        default_direction = torch.tensor([1.0, 0.0], device=self.device)
        current_lin_vel_direction[~non_zero_mask] = default_direction

        # 计算新的线速度向量（只在xy平面）
        new_lin_vel_vector = torch.zeros_like(self._asset.data.root_lin_vel_b)
        new_lin_vel_vector[:, :2] = clamped_lin_vel.unsqueeze(-1) * current_lin_vel_direction

        root_velocities = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocities[:, 0:3] = new_lin_vel_vector  # 线速度（z方向保持为0）
        root_velocities[:, 5] = clamped_ang_vel  # 角速度

        # 应用四元数变换
        root_velocities[:, 0:3] = math_utils.quat_apply(quat, root_velocities[:, 0:3])

        self._asset.write_root_link_velocity_to_sim(root_velocities)




class UAVBodyAction(BodyAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.UAVBodyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UAVBodyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_root_state[:, self._body_idx].clone()

        self.acc_limit = cfg.acc_limit

    def apply_actions(self):
        # 获取当前总速度
        current_vel = torch.norm(self._asset.data.root_lin_vel_b, dim=-1)
        quat = self._asset.data.root_link_quat_w

        # 目标速度向量（动作直接对应速度）
        target_vel_vector = self.processed_actions[:, 0:3]
        target_vel_magnitude = torch.norm(target_vel_vector, dim=-1)
        
        # 使用配置中的加速度限制
        acc_limit_per_step = self.cfg.acc_limit_per_step
        
        # 对总速度进行加速度限制 (基于PX4/Prometheus标准)
        # 限制速度变化率，确保平滑运动
        clamped_vel_magnitude = torch.clamp(
            target_vel_magnitude, 
            min=current_vel - acc_limit_per_step, 
            max=current_vel + acc_limit_per_step
        )

        # 计算速度方向（基于目标速度方向或当前速度方向）
        target_vel_direction = torch.zeros_like(target_vel_vector)
        non_zero_mask = target_vel_magnitude > 1e-6
        target_vel_direction[non_zero_mask] = target_vel_vector[non_zero_mask] / target_vel_magnitude[non_zero_mask].unsqueeze(-1)
        
        # 如果目标速度为零，使用当前速度方向或默认方向
        current_vel_direction = torch.zeros_like(self._asset.data.root_lin_vel_b)
        current_non_zero_mask = current_vel > 1e-6
        current_vel_direction[current_non_zero_mask] = self._asset.data.root_lin_vel_b[current_non_zero_mask] / current_vel[current_non_zero_mask].unsqueeze(-1)
        
        # 默认方向（向上）
        default_direction = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        target_vel_direction[~non_zero_mask] = current_vel_direction[~non_zero_mask]
        target_vel_direction[~non_zero_mask & ~current_non_zero_mask] = default_direction

        # 计算新的速度向量
        new_vel_vector = clamped_vel_magnitude.unsqueeze(-1) * target_vel_direction

        root_velocities = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocities[:, 0:3] = new_vel_vector

        # 应用四元数变换
        root_velocities[:, 0:3] = math_utils.quat_apply(quat, root_velocities[:, 0:3])

        self._asset.write_root_link_velocity_to_sim(root_velocities)






class ExternalForceTorqueAction(BodyAction):

    cfg: actions_cfg.ExternalForceTorqueActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.ExternalForceTorqueActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._body_id = self._asset.find_bodies("body")[0]
        self._robot_mass = self._asset.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor((0.0, 0.0, -9.81), device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

    def apply_actions(self):
        _actions = self.raw_actions.clone().clamp(-1.0, 1.0)

        _thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        _moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        _thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (_actions[:, 0] + 1.0) / 2.0
        _moment[:, 0, :] = self.cfg.moment_scale * _actions[:, 1:]


        self._asset.set_external_force_and_torque(_thrust, _moment, body_ids=self._body_id)





