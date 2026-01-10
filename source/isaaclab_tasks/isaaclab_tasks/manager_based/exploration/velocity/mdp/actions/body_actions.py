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
    """UGV速度模式动作：Policy输出目标速度，物理限制保证平滑和安全。
    
    控制流程：
    1. Policy输出 [-1,1] × scale = 目标速度 (线速度, 角速度)
    2. 加速度限制：平滑过渡到目标速度
    3. 速度上限：确保不超过最大速度
    
    差速驱动模型：
    - 线速度：沿机器人前向方向 (可正可负)
    - 角速度：绕z轴旋转
    """

    cfg: actions_cfg.UGVBodyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UGVBodyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_root_state[:, self._body_idx].clone()

    def process_actions(self, actions: torch.Tensor):
        """Override to apply different scales for linear and angular velocity."""
        self._raw_actions[:] = actions
        # 分别对线速度和角速度应用不同的scale
        self._processed_actions[:, 0] = actions[:, 0] * self.cfg.lin_scale
        self._processed_actions[:, 1] = actions[:, 1] * self.cfg.ang_scale

    def apply_actions(self):
        """Apply velocity commands with Nav2-style physical limits.
        
        速度模式 + 加速度限制 + 速度上限
        """
        # 获取当前速度 (body frame)
        current_lin_vel = self._asset.data.root_lin_vel_b[:, 0]  # 前向速度 (x方向)
        current_ang_vel = self._asset.data.root_ang_vel_b[:, 2]  # z轴角速度
        quat = self._asset.data.root_link_quat_w

        # 目标速度（速度模式：processed_actions 直接是目标速度）
        target_lin_vel = self._processed_actions[:, 0]
        target_ang_vel = self._processed_actions[:, 1]
        
        # 获取物理限制参数
        lin_acc = self.cfg.lin_acc_per_step
        ang_acc = self.cfg.ang_acc_per_step
        max_lin_vel = self.cfg.max_lin_vel
        max_ang_vel = self.cfg.max_ang_vel
        
        # ========== Step 1: 线速度加速度限制 ==========
        new_lin_vel = torch.clamp(
            target_lin_vel, 
            min=current_lin_vel - lin_acc, 
            max=current_lin_vel + lin_acc
        )
        
        # ========== Step 2: 线速度上限 ==========
        new_lin_vel = torch.clamp(new_lin_vel, min=-max_lin_vel, max=max_lin_vel)

        # ========== Step 3: 角速度加速度限制 ==========
        new_ang_vel = torch.clamp(
            target_ang_vel, 
            min=current_ang_vel - ang_acc, 
            max=current_ang_vel + ang_acc
        )
        
        # ========== Step 4: 角速度上限 ==========
        new_ang_vel = torch.clamp(new_ang_vel, min=-max_ang_vel, max=max_ang_vel)

        # 构建速度命令（差速驱动：前向速度 + 旋转）
        root_velocities = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocities[:, 0] = new_lin_vel  # 前向线速度 (body frame x)
        root_velocities[:, 5] = new_ang_vel  # z轴角速度

        # 从 body frame 转换到 world frame
        root_velocities[:, 0:3] = math_utils.quat_apply(quat, root_velocities[:, 0:3])

        self._asset.write_root_link_velocity_to_sim(root_velocities)




class UAVBodyAction(BodyAction):
    """UAV速度模式动作：Policy输出机体系目标速度，自动航向跟随。
    
    控制流程：
    1. Policy输出 [-1,1] × scale = 机体系目标速度 (vx, vy, vz)
    2. 加速度限制：平滑过渡到目标速度
    3. 速度上限：确保不超过最大速度
    4. 航向跟随：无人机自动朝向世界系水平速度方向 (yaw 控制)
    
    物理限制（基于PX4飞控）：
    - 水平方向：向量加速度限制 + 速度上限
    - 垂直方向：非对称加速度限制（上/下不同）+ 速度上限
    - 航向角速度：P控制器跟随速度方向
    
    部署时使用 Prometheus Move_mode=4 (XYZ_VEL_BODY)
    """

    cfg: actions_cfg.UAVBodyActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UAVBodyActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_root_state[:, self._body_idx].clone()

    def process_actions(self, actions: torch.Tensor):
        """Override to apply different scales for horizontal and vertical velocity."""
        self._raw_actions[:] = actions
        # 水平方向 (x, y) 使用 scale_hor
        self._processed_actions[:, 0] = actions[:, 0] * self.cfg.scale_hor
        self._processed_actions[:, 1] = actions[:, 1] * self.cfg.scale_hor
        # 垂直方向 (z) 使用 scale_z
        self._processed_actions[:, 2] = actions[:, 2] * self.cfg.scale_z

    def apply_actions(self):
        """Apply velocity commands with PX4-style physical limits + yaw following.
        
        速度模式 + 加速度限制 + 速度上限 + 航向跟随
        """
        # 获取当前速度 (body frame)
        current_vel = self._asset.data.root_lin_vel_b.clone()  # (num_envs, 3)
        current_ang_vel = self._asset.data.root_ang_vel_b[:, 2]  # z轴角速度
        quat = self._asset.data.root_link_quat_w

        # 目标速度向量（速度模式：processed_actions 直接是目标速度，机体系）
        target_vel = self.processed_actions[:, 0:3]  # (num_envs, 3)
        
        # 获取物理限制参数
        acc_hor = self.cfg.acc_hor_per_step    # 水平加速度限制 (向量magnitude)
        acc_up = self.cfg.acc_up_per_step      # 向上加速度限制
        acc_down = self.cfg.acc_down_per_step  # 向下加速度限制
        max_vel_hor = self.cfg.max_vel_hor     # 水平最大速度
        max_vel_z = self.cfg.max_vel_z         # 垂直最大速度
        
        new_vel = torch.zeros_like(current_vel)
        
        # ========== Step 1: 水平方向加速度限制 (向量限制) ==========
        delta_vel_hor = target_vel[:, :2] - current_vel[:, :2]  # (num_envs, 2)
        delta_magnitude = torch.norm(delta_vel_hor, dim=1, keepdim=True)  # (num_envs, 1)
        
        # 如果变化量超过加速度限制，按比例缩放
        acc_scale = torch.clamp(acc_hor / (delta_magnitude + 1e-6), max=1.0)  # (num_envs, 1)
        new_vel[:, :2] = current_vel[:, :2] + delta_vel_hor * acc_scale
        
        # ========== Step 2: 水平速度上限 (向量限制) ==========
        vel_hor_magnitude = torch.norm(new_vel[:, :2], dim=1, keepdim=True)  # (num_envs, 1)
        vel_scale = torch.clamp(max_vel_hor / (vel_hor_magnitude + 1e-6), max=1.0)
        new_vel[:, :2] = new_vel[:, :2] * vel_scale
        
        # ========== Step 3: 垂直方向加速度限制 (非对称) ==========
        new_vel[:, 2] = torch.clamp(
            target_vel[:, 2],
            min=current_vel[:, 2] - acc_down,  # 向下减速/加速
            max=current_vel[:, 2] + acc_up     # 向上加速
        )
        
        # ========== Step 4: 垂直速度上限 ==========
        new_vel[:, 2] = torch.clamp(new_vel[:, 2], min=-max_vel_z, max=max_vel_z)
        
        # ========== Step 5: 航向跟随 (yaw 自动朝向世界系速度方向) ==========
        # 将机体系速度转换到世界系
        vel_world = math_utils.quat_apply(quat, new_vel)
        vel_hor_world = vel_world[:, :2]  # 世界系水平速度
        vel_hor_mag = torch.norm(vel_hor_world, dim=1)
        
        # 只有当水平速度足够大时才跟随 (避免静止时抖动)
        vel_threshold = 0.3  # m/s
        has_velocity = vel_hor_mag > vel_threshold
        
        # 计算目标 yaw (世界系速度方向)
        target_yaw = torch.atan2(vel_hor_world[:, 1], vel_hor_world[:, 0])
        
        # 计算当前 yaw (从四元数提取)
        # quat: (w, x, y, z) 或 (x, y, z, w) - 需要确认格式
        # Isaac Sim 使用 (w, x, y, z) 格式
        current_yaw = math_utils.quat_to_euler(quat)[:, 2]  # 提取 yaw
        
        # 计算 yaw 误差 (wrap to [-pi, pi])
        yaw_error = target_yaw - current_yaw
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        
        # P控制器计算目标角速度
        yaw_rate_gain = self.cfg.yaw_rate_gain  # P控制器增益
        max_yaw_rate = self.cfg.max_yaw_rate    # 最大角速度 rad/s
        
        target_yaw_rate = yaw_rate_gain * yaw_error
        target_yaw_rate = torch.clamp(target_yaw_rate, -max_yaw_rate, max_yaw_rate)
        
        # 只在有速度时才应用航向控制
        target_yaw_rate = torch.where(has_velocity, target_yaw_rate, torch.zeros_like(target_yaw_rate))
        
        # 角速度加速度限制 (平滑)
        yaw_acc = self.cfg.yaw_acc_per_step  # 每步最大角速度变化
        new_yaw_rate = torch.clamp(
            target_yaw_rate,
            min=current_ang_vel - yaw_acc,
            max=current_ang_vel + yaw_acc
        )
        
        # ========== 构建速度命令 ==========
        root_velocities = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocities[:, 0:3] = new_vel                 # 机体系线速度
        root_velocities[:, 5] = new_yaw_rate              # z轴角速度

        # 从 body frame 转换到 world frame (只转换线速度)
        root_velocities[:, 0:3] = math_utils.quat_apply(quat, root_velocities[:, 0:3])

        self._asset.write_root_link_velocity_to_sim(root_velocities)






class ExternalForceTorqueAction(BodyAction):
    """直接推力+力矩控制 (无下层控制器)
    
    动作空间: [thrust, moment_x, moment_y, moment_z] (4维)
    用于端到端学习无人机控制
    """

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


class UAVVelocityWithDynamicsAction(BodyAction):
    """速度控制 + 真实动力学 (带下层控制器)
    
    策略输出: [vx, vy, vz, yaw_rate] (4维)
    - vx, vy, vz: 机体坐标系目标速度 (m/s)
    - yaw_rate: 目标航向角速度 (rad/s)
    
    下层控制器: 速度 → 姿态 → 推力+力矩
    
    相比 UAVBodyAction:
    - ✅ 有真实的惯性和动力学响应
    - ✅ 更接近真实无人机行为
    - ✅ 更好的 Sim-to-Real 迁移
    - ✅ 策略可控制航向
    
    控制流程:
    1. 目标速度 → PID → 目标姿态角 (pitch, roll)
    2. 目标姿态 → PID → 力矩 (mx, my)
    3. 目标 yaw_rate → PID → 力矩 (mz)
    4. 高度控制 → PID → 推力
    """

    cfg: actions_cfg.UAVVelocityWithDynamicsActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.UAVVelocityWithDynamicsActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        # 获取机体信息
        self._body_id = self._asset.find_bodies("body")[0]
        self._robot_mass = self._asset.root_physx_view.get_masses()[0].sum()
        self._gravity = 9.81
        self._robot_weight = (self._robot_mass * self._gravity).item()
        
        # 获取惯性矩 (用于力矩计算)
        # inertias 格式: [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] 或 [Ixx, Iyy, Izz]
        try:
            inertias = self._asset.root_physx_view.get_inertias()
            if inertias is not None and len(inertias) > 0:
                # 取主惯性矩 [Ixx, Iyy, Izz]
                self._inertia = inertias[0, :3].clone()  # shape: (3,)
                omni.log.info(f"UAV inertia: Ixx={self._inertia[0]:.4f}, Iyy={self._inertia[1]:.4f}, Izz={self._inertia[2]:.4f}")
            else:
                # 使用默认惯性 (基于 P450 估算)
                self._inertia = torch.tensor([0.01, 0.01, 0.02], device=self.device)
                omni.log.warn("Could not get inertia, using default values")
        except Exception as e:
            # 使用默认惯性
            self._inertia = torch.tensor([0.01, 0.01, 0.02], device=self.device)
            omni.log.warn(f"Failed to get inertia: {e}, using default values")
        
        omni.log.info(f"UAV mass: {self._robot_mass:.3f} kg, weight: {self._robot_weight:.3f} N")
        
        # PID 控制器状态
        self._vel_error_integral = torch.zeros(self.num_envs, 3, device=self.device)
        self._att_error_integral = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_vel_error = torch.zeros(self.num_envs, 3, device=self.device)
        self._last_att_error = torch.zeros(self.num_envs, 3, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        """处理策略输出: [vx, vy, vz, yaw_rate]"""
        self._raw_actions[:] = actions
        # 水平方向 (x, y) 使用 scale_hor
        self._processed_actions[:, 0] = actions[:, 0] * self.cfg.scale_hor
        self._processed_actions[:, 1] = actions[:, 1] * self.cfg.scale_hor
        # 垂直方向 (z) 使用 scale_z
        self._processed_actions[:, 2] = actions[:, 2] * self.cfg.scale_z
        # 航向角速度 使用 scale_yaw
        self._processed_actions[:, 3] = actions[:, 3] * self.cfg.scale_yaw

    def apply_actions(self):
        """将目标速度和yaw_rate转换为推力和力矩"""
        # 获取当前状态
        current_vel_w = self._asset.data.root_lin_vel_w  # 世界系速度
        current_vel_b = self._asset.data.root_lin_vel_b  # 机体系速度
        current_ang_vel_b = self._asset.data.root_ang_vel_b  # 机体系角速度
        quat = self._asset.data.root_link_quat_w  # 四元数 (w, x, y, z)
        
        # 目标速度 (机体系) 和 目标航向角速度
        target_vel_b = self.processed_actions[:, 0:3].clone()
        target_yaw_rate = self.processed_actions[:, 3]  # 策略输出的航向角速度
        
        # ========== Step 0: 速度限制 (与 PX4 一致) ==========
        # 水平速度限制 (向量方式)
        vel_hor = target_vel_b[:, :2]
        vel_hor_mag = torch.norm(vel_hor, dim=1, keepdim=True)
        vel_hor_scale = torch.clamp(self.cfg.max_vel_hor / (vel_hor_mag + 1e-6), max=1.0)
        target_vel_b[:, :2] = vel_hor * vel_hor_scale
        
        # 垂直速度限制
        target_vel_b[:, 2] = torch.clamp(target_vel_b[:, 2], -self.cfg.max_vel_z, self.cfg.max_vel_z)
        
        # 航向角速度限制
        target_yaw_rate = torch.clamp(target_yaw_rate, -self.cfg.max_yaw_rate, self.cfg.max_yaw_rate)
        
        # ========== Step 1: 速度控制 → 目标姿态 ==========
        # 速度误差 (机体系)
        vel_error = target_vel_b - current_vel_b
        
        # PID 计算
        dt = self.cfg.control_dt
        self._vel_error_integral += vel_error * dt
        self._vel_error_integral = torch.clamp(self._vel_error_integral, -1.0, 1.0)  # 防止积分饱和
        vel_error_derivative = (vel_error - self._last_vel_error) / dt
        self._last_vel_error = vel_error.clone()
        
        # 速度 PID 输出 → 期望加速度
        Kp_vel = self.cfg.vel_kp
        Ki_vel = self.cfg.vel_ki
        Kd_vel = self.cfg.vel_kd
        
        desired_acc = Kp_vel * vel_error + Ki_vel * self._vel_error_integral + Kd_vel * vel_error_derivative
        
        # ========== Step 1.5: 加速度限制 (与 PX4 一致) ==========
        # 水平加速度限制 (向量方式)
        acc_hor = desired_acc[:, :2]
        acc_hor_mag = torch.norm(acc_hor, dim=1, keepdim=True)
        acc_hor_scale = torch.clamp(self.cfg.max_acc_hor / (acc_hor_mag + 1e-6), max=1.0)
        desired_acc[:, :2] = acc_hor * acc_hor_scale
        
        # 垂直加速度限制
        desired_acc[:, 2] = torch.clamp(desired_acc[:, 2], -self.cfg.max_acc_z, self.cfg.max_acc_z)
        
        # 期望加速度 → 期望姿态角 (小角度近似)
        # ax ≈ g * tan(pitch) ≈ g * pitch
        # ay ≈ -g * tan(roll) ≈ -g * roll
        target_pitch = torch.clamp(desired_acc[:, 0] / self._gravity, -0.5, 0.5)  # 约 ±30°
        target_roll = torch.clamp(-desired_acc[:, 1] / self._gravity, -0.5, 0.5)
        
        # ========== Step 2: 姿态控制 → 力矩 ==========
        # 获取当前姿态角 (从四元数提取 roll, pitch, yaw)
        current_euler = math_utils.quat_to_euler(quat)  # (roll, pitch, yaw)
        current_roll = current_euler[:, 0]
        current_pitch = current_euler[:, 1]
        
        # 姿态误差 (roll, pitch)
        roll_error = target_roll - current_roll
        pitch_error = target_pitch - current_pitch
        
        # 航向控制: 使用策略输出的 yaw_rate
        # yaw_error = target_yaw_rate - current_yaw_rate
        current_yaw_rate = current_ang_vel_b[:, 2]
        yaw_rate_error = target_yaw_rate - current_yaw_rate
        
        # 组合姿态误差 (roll, pitch 用角度误差; yaw 用角速度误差)
        att_error = torch.stack([roll_error, pitch_error, yaw_rate_error], dim=1)
        
        # 姿态 PID
        self._att_error_integral += att_error * dt
        self._att_error_integral = torch.clamp(self._att_error_integral, -0.5, 0.5)
        att_error_derivative = (att_error - self._last_att_error) / dt
        self._last_att_error = att_error.clone()
        
        Kp_att = self.cfg.att_kp
        Ki_att = self.cfg.att_ki
        Kd_att = self.cfg.att_kd
        
        # 力矩 = (PID输出 - 角速度阻尼) × 惯性矩
        # 物理公式: moment = inertia × angular_acceleration
        angular_acc_cmd = (Kp_att * att_error + Ki_att * self._att_error_integral + 
                           Kd_att * att_error_derivative - self.cfg.ang_vel_damping * current_ang_vel_b)
        
        # 将惯性矩扩展到所有环境
        inertia = self._inertia.unsqueeze(0).expand(self.num_envs, 3)  # (num_envs, 3)
        
        # 力矩 = 惯性 × 角加速度 × 缩放因子
        moment = angular_acc_cmd * inertia * self.cfg.moment_scale
        
        # ========== Step 3: 高度控制 → 推力 ==========
        # 目标 z 速度 → 推力
        vz_error = target_vel_b[:, 2] - current_vel_b[:, 2]
        
        # 基础悬停推力 + 速度补偿
        hover_thrust = self._robot_weight / torch.cos(current_roll) / torch.cos(current_pitch)
        thrust_compensation = self.cfg.thrust_kp * vz_error * self._robot_mass
        total_thrust = hover_thrust + thrust_compensation
        total_thrust = torch.clamp(total_thrust, 0, self._robot_weight * 2.0)  # 限制推力
        
        # ========== 应用力和力矩 ==========
        _thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        _moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        
        _thrust[:, 0, 2] = total_thrust
        _moment[:, 0, :] = moment
        
        self._asset.set_external_force_and_torque(_thrust, _moment, body_ids=self._body_id)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """重置 PID 状态"""
        super().reset(env_ids)
        if env_ids is not None:
            self._vel_error_integral[env_ids] = 0.0
            self._att_error_integral[env_ids] = 0.0
            self._last_vel_error[env_ids] = 0.0
            self._last_att_error[env_ids] = 0.0





