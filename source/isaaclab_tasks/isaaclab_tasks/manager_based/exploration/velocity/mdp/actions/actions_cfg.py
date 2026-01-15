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
    scale_hor: float = 1.0   # 水平速度缩放 (m/s) - action=1 → 1 m/s
    scale_z: float = 2.0     # 垂直速度缩放 (m/s) - action=1 → 2 m/s
    
    # 速度限制配置 (基于PX4飞控标准: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX)
    max_vel_hor: float = 1.0   # 水平最大速度 (m/s)
    max_vel_z: float = 2.0     # 垂直最大速度 (m/s)
    
    # 加速度限制配置 (基于PX4飞控标准: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    acc_hor: float = 2.0    # 水平最大加速度 (m/s²) - 向量限制
    acc_up: float = 3.0     # 向上最大加速度 (m/s²)
    acc_down: float = 2.0   # 向下最大加速度 (m/s²) - 安全限制
    
    # ========== 航向跟随配置 ==========
    # 无人机自动朝向世界系水平速度方向
    yaw_rate_gain: float = 2.0    # 航向P控制器增益
    max_yaw_rate: float = 0.5     # 最大角速度 (rad/s) ≈ 29°/s
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
class UAVBodyActionWithYawCfg(BodyActionCfg):
    """UAV速度模式动作配置 + 策略可控 yaw（质点模型）。
    
    相比 UAVBodyActionCfg：
    - 策略直接输出目标 yaw 角度（而非 yaw_rate 或自动跟随）
    - 4维动作空间
    - 质点模型，无动力学，训练更快
    
    动作空间: [vx, vy, vz, yaw] (4维)
    - vx, vy, vz: 机体系目标速度 (m/s)
    - yaw: 目标航向角，相对于当前朝向的偏移量 (rad)
    
    部署时使用 Prometheus Move_mode=4 (XYZ_VEL_BODY) + yaw
    """

    use_default_offset: bool = True
    action_dim = 4  # [vx, vy, vz, yaw_offset]
    
    # 动作缩放 (与 velocity_env_cfg.py 保持一致)
    scale_hor: float = 1.0     # 水平速度缩放 (m/s): action=1 → 1 m/s
    scale_z: float = 2.0       # 垂直速度缩放 (m/s): action=1 → 2 m/s
    scale_yaw: float = 1.57    # 航向角缩放 (rad): action=1 → π/2 rad (90°)
    
    # 速度限制 (PX4: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX_UP/DN)
    max_vel_hor: float = 1.0   # 水平最大速度 (m/s)
    max_vel_up: float = 2.0    # 上升最大速度 (m/s)
    max_vel_down: float = 1.0  # 下降最大速度 (m/s)
    max_yaw_rate: float = 0.5  # 最大航向角速度 (rad/s) ≈ 29°/s
    
    # 航向 P 控制器增益
    yaw_p_gain: float = 2.0    # P 控制器增益: yaw_rate = Kp * yaw_error
    
    # 加速度限制 (PX4: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    acc_hor: float = 2.0    # 水平最大加速度 (m/s²) - 向量限制
    acc_up: float = 3.0     # 向上最大加速度 (m/s²)
    acc_down: float = 2.0   # 向下最大加速度 (m/s²)
    yaw_acc: float = 3.0    # 角加速度限制 (rad/s²)
    
    # 时间步长配置
    sim_dt: float = 0.01
    decimation: int = 10
    
    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation
    
    @property
    def acc_hor_per_step(self) -> float:
        return self.acc_hor * self.control_dt
    
    @property
    def acc_up_per_step(self) -> float:
        return self.acc_up * self.control_dt
    
    @property
    def acc_down_per_step(self) -> float:
        return self.acc_down * self.control_dt
    
    @property
    def yaw_acc_per_step(self) -> float:
        return self.yaw_acc * self.control_dt

    class_type: type[ActionTerm] = body_actions.UAVBodyActionWithYaw


@configclass
class UAVBodyActionAutoYawCfg(BodyActionCfg):
    """UAV速度模式动作配置 + 航向自动朝向目标（质点模型）。
    
    方案四：最简单的控制模式
    - 3维动作空间：[vx, vy, vz]
    - 航向自动朝向目标点，策略不控制 yaw
    - 更容易学习，适合初期训练
    
    动作空间: [vx, vy, vz] (3维)
    - vx, vy, vz: 机体坐标系目标速度 (m/s)
    
    航向控制：
    target_yaw = direction_to_goal  # 始终朝向目标
    使用 P 控制器将 target_yaw 转换为 yaw_rate
    
    部署时使用 Prometheus Move_mode=4 (XYZ_VEL_BODY) + auto yaw
    """

    use_default_offset: bool = True
    action_dim = 3  # [vx, vy, vz] - 无 yaw 动作
    
    # 动作缩放 (与其他 UAV 配置保持一致)
    scale_hor: float = 1.0     # 水平速度缩放 (m/s): action=1 → 1 m/s
    scale_z: float = 2.0       # 垂直速度缩放 (m/s): action=1 → 2 m/s
    
    # 速度限制 (PX4: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX_UP/DN)
    max_vel_hor: float = 1.0   # 水平最大速度 (m/s)
    max_vel_up: float = 2.0    # 上升最大速度 (m/s)
    max_vel_down: float = 1.0  # 下降最大速度 (m/s)
    max_yaw_rate: float = 0.5  # 最大航向角速度 (rad/s) ≈ 29°/s
    
    # 航向 P 控制器增益
    yaw_p_gain: float = 0.5    # P 控制器增益: yaw_rate = Kp * yaw_error
    
    # 加速度限制 (PX4: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    acc_hor: float = 2.0    # 水平最大加速度 (m/s²) - 向量限制
    acc_up: float = 3.0     # 向上最大加速度 (m/s²)
    acc_down: float = 2.0   # 向下最大加速度 (m/s²)
    yaw_acc: float = 3.0    # 角加速度限制 (rad/s²)
    
    # 时间步长配置
    sim_dt: float = 0.01
    decimation: int = 10
    
    @property
    def control_dt(self) -> float:
        return self.sim_dt * self.decimation
    
    @property
    def acc_hor_per_step(self) -> float:
        return self.acc_hor * self.control_dt
    
    @property
    def acc_up_per_step(self) -> float:
        return self.acc_up * self.control_dt
    
    @property
    def acc_down_per_step(self) -> float:
        return self.acc_down * self.control_dt
    
    @property
    def yaw_acc_per_step(self) -> float:
        return self.yaw_acc * self.control_dt

    class_type: type[ActionTerm] = body_actions.UAVBodyActionAutoYaw


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
    scale_hor: float = 1.0     # 水平速度缩放 (m/s): action=1 → 1 m/s
    scale_z: float = 2.0       # 垂直速度缩放 (m/s): action=1 → 2 m/s
    scale_yaw: float = 1.57    # 航向角速度缩放 (rad/s): action=1 → π/2 rad/s
    
    # 速度限制 (PX4: MPC_XY_VEL_MAX, MPC_Z_VEL_MAX_UP/DN)
    max_vel_hor: float = 1.0   # 水平最大速度 (m/s)
    max_vel_up: float = 2.0    # 上升最大速度 (m/s)
    max_vel_down: float = 1.0  # 下降最大速度 (m/s)
    max_yaw_rate: float = 0.5  # 最大航向角速度 (rad/s) ≈ 29°/s
    
    # 加速度限制 (PX4: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
    max_acc_hor: float = 2.0   # 水平最大加速度 (m/s²)
    max_acc_z: float = 3.0     # 垂直最大加速度 (m/s²)
    
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
    
    # ========== 速度控制 PID ==========
    vel_kp: float = 0.5     # 速度 P 增益
    vel_ki: float = 0.02    # 小积分项，消除稳态误差
    vel_kd: float = 0.0     # 关闭微分
    
    # ========== 姿态控制 PID ==========
    att_kp: float = 4.0     # 姿态 P 增益
    att_ki: float = 0.01    # 小积分项，消除稳态误差
    att_kd: float = 0.0     # 关闭微分
    
    # ========== 角速度阻尼 ==========
    ang_vel_damping: float = 1.0   # 角速度阻尼 (适中)
    
    # ========== 推力控制 ==========
    thrust_kp: float = 3.0  # 推力 P 增益
    
    # ========== 力矩缩放因子 ==========
    # 必须足够大才能产生姿态变化！
    moment_scale: float = 1.0   # 力矩缩放 (恢复正常)
    
    class_type: type[ActionTerm] = body_actions.UAVVelocityWithDynamicsAction

