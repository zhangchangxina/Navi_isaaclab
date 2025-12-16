# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.controllers import DifferentialIKControllerCfg, OperationalSpaceControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from . import body_actions



##
# Body actions.
##


@configclass
class BodyActionCfg(ActionTermCfg):
    """Configuration for the base joint action term.

    See :class:`JointAction` for more details.
    """

    body_name: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | list[float] | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""
    # action_dim: int = MISSING
    action_dim = 6


@configclass
class UGVBodyActionCfg(BodyActionCfg):

    use_default_offset: bool = True
    action_dim = 2
    
    # 加速度限制配置 (基于Nav2标准)
    lin_acc_limit: float = 2.5  # 线速度加速度限制 (m/s²)
    ang_acc_limit: float = 3.2  # 角速度加速度限制 (rad/s²)
    
    # 时间步长配置
    sim_dt: float = 0.01  # 仿真时间步 (秒)
    decimation: int = 10  # 控制频率 (步数)
    
    # 计算实际控制时间步
    @property
    def control_dt(self) -> float:
        """实际控制时间步 (秒)"""
        return self.sim_dt * self.decimation
    
    # 计算每步速度变化限制
    @property
    def lin_acc_limit_per_step(self) -> float:
        """线速度每步变化限制 (m/s)"""
        return self.lin_acc_limit * self.control_dt
    
    @property
    def ang_acc_limit_per_step(self) -> float:
        """角速度每步变化限制 (rad/s)"""
        return self.ang_acc_limit * self.control_dt

    class_type: type[ActionTerm] = body_actions.UGVBodyAction


@configclass
class UAVBodyActionCfg(BodyActionCfg):

    use_default_offset: bool = True
    action_dim = 3
    
    # 加速度限制配置 (基于PX4/Prometheus标准)
    acc_limit: float = 6.5  # 总速度加速度限制 (m/s²)
    
    # 时间步长配置
    sim_dt: float = 0.01  # 仿真时间步 (秒)
    decimation: int = 10  # 控制频率 (步数)
    
    # 计算实际控制时间步
    @property
    def control_dt(self) -> float:
        """实际控制时间步 (秒)"""
        return self.sim_dt * self.decimation
    
    # 计算每步速度变化限制
    @property
    def acc_limit_per_step(self) -> float:
        """总速度每步变化限制 (m/s)"""
        return self.acc_limit * self.control_dt

    class_type: type[ActionTerm] = body_actions.UAVBodyAction


@configclass
class ExternalForceTorqueActionCfg(BodyActionCfg):
    """Configuration for the bounded joint position action term.

    See :class:`JointPositionToLimitsAction` for more details.
    """

    # joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    scale: float | dict[str, float] = 1.0
    """Scale factor for the action (float or dict of regex expressions). Defaults to 1.0."""
    offset: float | dict[str, float] = 0.0
    """Offset factor for the action (float or dict of regex expressions). Defaults to 0.0."""
    preserve_order: bool = False
    """Whether to preserve the order of the joint names in the action output. Defaults to False."""

    class_type: type[ActionTerm] = body_actions.ExternalForceTorqueAction

    body_name: list[str] = MISSING
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01

