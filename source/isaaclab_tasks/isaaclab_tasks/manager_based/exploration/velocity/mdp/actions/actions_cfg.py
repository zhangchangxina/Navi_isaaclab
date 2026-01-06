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
    """UGV速度模式动作配置（差速驱动）。
    
    速度模式：Policy输出[-1,1] × scale = 目标速度
    物理限制：加速度限制 + 速度上限
    
    动作维度：2 (线速度, 角速度)
    """

    use_default_offset: bool = True
    action_dim = 2
    
    # 动作缩放（分开设置线速度和角速度）
    lin_scale: float = 1.0   # 线速度缩放 (m/s)
    ang_scale: float = 1.5   # 角速度缩放 (rad/s)
    
    # 速度限制配置 (基于Nav2标准)
    max_lin_vel: float = 1.0   # 最大线速度 (m/s)
    max_ang_vel: float = 1.5   # 最大角速度 (rad/s)
    
    # 加速度限制配置 (基于Nav2标准)
    lin_acc: float = 2.5    # 线加速度限制 (m/s²)
    ang_acc: float = 3.2    # 角加速度限制 (rad/s²)
    
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
    def lin_acc_per_step(self) -> float:
        """线速度每步变化限制 (m/s)"""
        return self.lin_acc * self.control_dt
    
    @property
    def ang_acc_per_step(self) -> float:
        """角速度每步变化限制 (rad/s)"""
        return self.ang_acc * self.control_dt

    class_type: type[ActionTerm] = body_actions.UGVBodyAction


@configclass
class UAVBodyActionCfg(BodyActionCfg):
    """UAV速度模式动作配置。
    
    速度模式：Policy输出[-1,1] × scale = 目标速度
    物理限制：加速度限制 + 速度上限
    """

    use_default_offset: bool = True
    action_dim = 3
    
    # 动作缩放（分开设置水平和垂直）
    scale_hor: float = 3.0   # 水平速度缩放 (m/s) - action=1 → 3 m/s
    scale_z: float = 2.0     # 垂直速度缩放 (m/s) - action=1 → 2 m/s
    
    # 速度限制配置 (基于PX4飞控标准: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX)
    max_vel_hor: float = 3.0   # 水平最大速度 (m/s)
    max_vel_z: float = 2.0     # 垂直最大速度 (m/s)
    
    # 加速度限制配置 (基于PX4飞控标准: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    acc_hor: float = 3.0    # 水平最大加速度 (m/s²) - 向量限制
    acc_up: float = 3.0     # 向上最大加速度 (m/s²)
    acc_down: float = 2.0   # 向下最大加速度 (m/s²) - 安全限制
    
    # 时间步长配置
    sim_dt: float = 0.01  # 仿真时间步 (秒)
    decimation: int = 10  # 控制频率 (步数)
    
    # 计算实际控制时间步
    @property
    def control_dt(self) -> float:
        """实际控制时间步 (秒)"""
        return self.sim_dt * self.decimation
    
    # 计算每步水平速度变化限制
    @property
    def acc_hor_per_step(self) -> float:
        """水平方向每步速度变化限制 (m/s)"""
        return self.acc_hor * self.control_dt
    
    # 计算每步向上速度变化限制
    @property
    def acc_up_per_step(self) -> float:
        """向上每步速度变化限制 (m/s)"""
        return self.acc_up * self.control_dt
    
    # 计算每步向下速度变化限制
    @property
    def acc_down_per_step(self) -> float:
        """向下每步速度变化限制 (m/s)"""
        return self.acc_down * self.control_dt

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

