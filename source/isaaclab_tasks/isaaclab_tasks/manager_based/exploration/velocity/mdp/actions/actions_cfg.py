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
    """UAV速度模式动作配置 (机体系速度 + 航向跟随)。
    
    控制模式：
    - Policy输出[-1,1] × scale = 机体系目标速度 (vx_body, vy_body, vz_body)
    - 航向自动跟随：无人机自动朝向世界系水平速度方向
    
    物理限制：加速度限制 + 速度上限 + 角速度限制
    
    部署时使用 Prometheus Move_mode=4 (XYZ_VEL_BODY)
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
    
    # ========== 航向跟随配置 ==========
    # 无人机自动朝向世界系水平速度方向
    yaw_rate_gain: float = 2.0    # 航向P控制器增益
    max_yaw_rate: float = 1.5     # 最大角速度 (rad/s) ≈ 86°/s
    yaw_acc: float = 3.0          # 角加速度限制 (rad/s²)
    
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
    
    # 计算每步角速度变化限制
    @property
    def yaw_acc_per_step(self) -> float:
        """角速度每步变化限制 (rad/s)"""
        return self.yaw_acc * self.control_dt

    class_type: type[ActionTerm] = body_actions.UAVBodyAction


@configclass
class ExternalForceTorqueActionCfg(BodyActionCfg):
    """直接推力+力矩控制配置 (4维动作空间)
    
    动作: [thrust, moment_x, moment_y, moment_z]
    """

    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = False

    class_type: type[ActionTerm] = body_actions.ExternalForceTorqueAction

    body_name: list[str] = MISSING
    thrust_to_weight: float = 1.9
    moment_scale: float = 0.01


@configclass
class UAVVelocityWithDynamicsActionCfg(BodyActionCfg):
    """速度控制 + 真实动力学配置 (带下层 PID 控制器)
    
    策略输出: [vx, vy, vz, yaw_rate] (4维)
    - vx, vy, vz: 机体坐标系目标速度 (m/s)
    - yaw_rate: 目标航向角速度 (rad/s)
    
    内部控制器将速度命令转换为推力+力矩
    
    优点:
    - 完整的 4 自由度控制
    - 有真实的动力学响应 (惯性、延迟)
    - 更好的 Sim-to-Real 迁移
    
    部署时使用 Prometheus Move_mode=4 (XYZ_VEL_BODY) + yaw_rate
    """

    use_default_offset: bool = True
    action_dim = 4  # [vx, vy, vz, yaw_rate]
    
    # 动作缩放
    scale_hor: float = 3.0     # 水平速度缩放 (m/s): action=1 → 3 m/s
    scale_z: float = 2.0       # 垂直速度缩放 (m/s): action=1 → 2 m/s
    scale_yaw: float = 1.5     # 航向角速度缩放 (rad/s): action=1 → 1.5 rad/s ≈ 86°/s
    
    # 速度限制 (PX4: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX_UP/DN)
    max_vel_hor: float = 3.0   # 水平最大速度 (m/s)
    max_vel_z: float = 2.0     # 垂直最大速度 (m/s)
    max_yaw_rate: float = 1.5  # 最大航向角速度 (rad/s)
    
    # 加速度限制 (PX4: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    max_acc_hor: float = 3.0   # 水平最大加速度 (m/s²), PX4 默认 5.0
    max_acc_z: float = 3.0     # 垂直最大加速度 (m/s²), PX4 默认 4.0
    
    # 时间步长
    sim_dt: float = 0.01
    decimation: int = 10
    
    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation
    
    # ========== 下层控制器 PID 参数 ==========
    # 参考 PX4 默认参数，确保 Sim-to-Real 一致性
    # PX4 参数文档: https://docs.px4.io/main/en/advanced_config/parameter_reference.html
    #
    # 重要：PX4 使用级联控制 (姿态 → 角速度 → 力矩)
    # 我们的简化实现跳过角速度环，所以需要调低增益 + 增加阻尼
    
    # 速度控制 PID (对应 PX4 MPC_XY_VEL_*, MPC_Z_VEL_*)
    # PX4 默认: P=0.95, I=0.05, D=0.02 (水平), P=0.4, I=0.15, D=0.0 (垂直)
    vel_kp: float = 0.95    # 速度 P 增益 (PX4: MPC_XY_VEL_P_ACC)
    vel_ki: float = 0.05    # 速度 I 增益 (PX4: MPC_XY_VEL_I_ACC)
    vel_kd: float = 0.02    # 速度 D 增益 (PX4: MPC_XY_VEL_D_ACC)
    
    # 姿态控制 P (对应 PX4 MC_ROLL_P × MC_ROLLRATE_P 的等效增益)
    # PX4: 6.5 × 0.15 ≈ 1.0，但因为我们跳过角速度环，需要更保守
    # 降低增益 + 提高阻尼 来补偿缺失的角速度控制环
    att_kp: float = 2.5     # 姿态 P 增益 (降低，原 6.5 太激进)
    att_ki: float = 0.0     # 姿态控制通常不用 I
    att_kd: float = 0.0     # 姿态控制通常不用 D (用角速度阻尼代替)
    
    # 角速度阻尼 (模拟 PX4 角速度控制环的效果)
    # 增加阻尼来抑制旋转过冲，防止翻转
    # 这是防止翻飞机的关键参数！
    ang_vel_damping: float = 0.8   # 增大阻尼 (原 0.15 太小)
    
    # 推力控制 (对应 PX4 MPC_Z_VEL_P_ACC)
    # PX4 默认: 4.0
    thrust_kp: float = 4.0  # 推力 P 增益
    
    # 力矩缩放因子
    # 物理公式: moment = inertia × angular_acceleration × moment_scale
    # 惯性矩自动从模型读取，此参数用于微调
    # 降低到 0.5 作为额外的安全系数
    moment_scale: float = 0.5   # 更保守的力矩 (原 1.0)
    
    class_type: type[ActionTerm] = body_actions.UAVVelocityWithDynamicsAction

