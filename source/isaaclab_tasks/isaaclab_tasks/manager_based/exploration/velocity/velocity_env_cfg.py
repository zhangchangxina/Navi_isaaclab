# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from enum import Enum

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, LOCAL_ASSET_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

import isaaclab_tasks.manager_based.exploration.velocity.mdp as mdp

##
# Robot type configuration
##

class RobotType(Enum):
    """Robot type enumeration."""
    UGV = "ugv"  # Ground vehicle
    UAV = "uav"  # Aerial vehicle

# 默认机器人类型
DEFAULT_ROBOT_TYPE = RobotType.UGV

##
# Pre-defined configs
##
from isaaclab.terrains.config.forest import FOREST_TERRAINS_CFG, PLANE_TERRAIN_CFG  # isort: skip
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##



@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    plane = TerrainImporterCfg(
        prim_path="/World/ground/plane",
        terrain_type="generator",
        terrain_generator=PLANE_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{LOCAL_ASSET_DIR}/Materials/Natural/Grass_Cut.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        # Debug visualization is useful for interactive play, but can break headless training.
        debug_vis=False,
    )


    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground/forest",
        terrain_type="generator",
        terrain_generator=FOREST_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{LOCAL_ASSET_DIR}/Materials/Wood/Oak.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        # Debug visualization is useful for interactive play, but can break headless training.
        debug_vis=False,
    )


    # robots
    robot: ArticulationCfg = MISSING

    lidar_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        update_period=1 / 60,
        # MID360 安装位置和姿态 (参考 SU17 官方文档)
        # 位置: x=0.13m(前), y=0, z=0.23m(上)
        # 姿态: 15° 前倾 (pitch), 四元数 (w,x,y,z) = (0.9914, 0, 0.1305, 0)
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.13, 0, 0.23),
            rot=(0.9914, 0, 0.1305, 0),  # 15° pitch forward tilt
        ),
        mesh_prim_paths=["/World/ground/forest"],
        ray_alignment='yaw',
        # Debug visualization is useful for interactive play, but can break headless training.
        debug_vis=False,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(180.0, 180.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # UGV: 0.22-0.26 m/s, UAV: 2 m/s
            lin_vel_x=(-0.25, 0.25), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
    )


    # 1.在障碍物区域边界随机发布目标点
    # pose_command = mdp.UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=False,
    #     resampling_time_range=(180.0, 180.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(25.5, 25.5), pos_y=(-25.0, 25.0), heading=(1.57, 4.71)),
    # )

    # 2.在障碍物区域内部随机发布目标点 (3D版本，高度随机1-5m)
    pose_command = mdp.TerrainBasedPoseCommandCfg(
        asset_name="robot",
        resampling_time_range=(180.0, 180.0),
        debug_vis=True,
        ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(pos_z=(1.0, 5.0)),  # 高度随机1-5m
        offset_z=0.0  # 高度完全由pos_z控制
    )

    # 3.在右上角发布目标点 (3D版本，高度随机1-5m)
    # pose_command = mdp.TerrainBasedPoseCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(180.0, 180.0),
    #     debug_vis=True,
    #     ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(pos_z=(1.0, 5.0)),  # 高度随机1-5m
    #     offset_z=0.0
    # )

    
    # 3.在右上角发布目标点
    # pose_command = mdp.UniformPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=False,
    #     resampling_time_range=(180.0, 180.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(22.0, 22.0), pos_y=(22.0, 22.0), heading=(1.57, 4.71)),
    # )


    traj_command = mdp.TrajectoryVisCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.1, 0.1),
        max_length=1000,
        threshold=1.0,  # 增大阈值，只有重置时（位置跳跃>10m）才清除轨迹
    )


# --------------------------------------------------------
# UGV 物理参数配置 (仿真加速版)
# 基于 Turtlebot3 Burger，速度提高约2倍以适应大地图训练
# 训练成功后可逐步降低速度进行 fine-tune
# --------------------------------------------------------
# 速度模式：Policy输出[-1,1] × scale = 目标速度
_UGV_MAX_LIN_VEL = 0.5      # 最大线速度 (m/s) - 仿真加速，约2倍真实速度
_UGV_MAX_ANG_VEL = 2.0      # 最大角速度 (rad/s) - 略低于真实值，更稳定
_UGV_CONTROL_DT = 0.1       # 控制周期 (s) = sim_dt(0.01) * decimation(10)

# 加速度限制 (适配加速后的速度)
_UGV_LIN_ACC = 1.0          # 线加速度限制 (m/s²)
_UGV_ANG_ACC = 2.0          # 角加速度限制 (rad/s²)

# 动作缩放 (速度模式：action=1.0 对应最大速度)
_UGV_LIN_ACTION_SCALE = _UGV_MAX_LIN_VEL  # 0.5
_UGV_ANG_ACTION_SCALE = _UGV_MAX_ANG_VEL  # 2.0


# --------------------------------------------------------
# UAV 物理参数配置 (Move out of ActionsCfg)
# 基于 PX4 飞控标准参数
# --------------------------------------------------------
# 速度模式：Policy输出[-1,1] × scale = 目标速度
_UAV_MAX_VEL_HOR = 3.0      # 水平最大速度 (m/s)
_UAV_MAX_VEL_Z = 2.0        # 垂直最大速度 (m/s) - 通常比水平慢
_UAV_CONTROL_DT = 0.1       # 控制周期

# 向量加速度限制 (基于PX4标准: MPC_ACC_HOR_MAX, MPC_ACC_UP_MAX, MPC_ACC_DOWN_MAX)
_UAV_ACC_HOR = 3.0          # 水平最大加速度 (m/s²) - 向量限制，任意方向一致
_UAV_ACC_UP = 3.0           # 向上最大加速度 (m/s²)
_UAV_ACC_DOWN = 2.0         # 向下最大加速度 (m/s²) - 安全限制，防止下降过快

@configclass
class ActionsCfg:
    """Action configuration."""
    
    ugv_action = mdp.UGVBodyActionCfg(
        asset_name="robot", 
        body_name=["body"],
        lin_scale=_UGV_LIN_ACTION_SCALE,   # 速度模式：action=1 → 线速度
        ang_scale=_UGV_ANG_ACTION_SCALE,   # 速度模式：action=1 → 角速度
        max_lin_vel=_UGV_MAX_LIN_VEL,      # 最大线速度限制
        max_ang_vel=_UGV_MAX_ANG_VEL,      # 最大角速度限制
        lin_acc=_UGV_LIN_ACC,              # 线加速度限制
        ang_acc=_UGV_ANG_ACC,              # 角加速度限制
    )

    # ============================================================
    # UAV Action 选择 (三选一)
    # ============================================================
    
    # 方案 1: 质点模型 - 直接写入速度 (无动力学, 训练快)
    # uav_action = mdp.UAVBodyActionCfg(
    #     asset_name="robot", 
    #     body_name=["body"],
    #     scale_hor=_UAV_MAX_VEL_HOR,
    #     scale_z=_UAV_MAX_VEL_Z,
    #     max_vel_hor=_UAV_MAX_VEL_HOR,
    #     max_vel_z=_UAV_MAX_VEL_Z,
    #     acc_hor=_UAV_ACC_HOR,
    #     acc_up=_UAV_ACC_UP,
    #     acc_down=_UAV_ACC_DOWN,
    # )
    
    # 方案 2: 策略输出速度+yaw_rate + 仿真用力矩 (有动力学, 推荐!)
    # 动作空间: [vx, vy, vz, yaw_rate] (4维)
    # PID 参数匹配 PX4 默认值，确保 Sim-to-Real 一致性
    uav_action = mdp.UAVVelocityWithDynamicsActionCfg(
        asset_name="robot", 
        body_name=["body"],
        scale_hor=_UAV_MAX_VEL_HOR,    # 水平速度缩放：action=1 → 3 m/s
        scale_z=_UAV_MAX_VEL_Z,        # 垂直速度缩放：action=1 → 2 m/s
        scale_yaw=1.5,                 # 航向角速度缩放：action=1 → 1.5 rad/s
        max_vel_hor=_UAV_MAX_VEL_HOR,  # 水平最大速度限制
        max_vel_z=_UAV_MAX_VEL_Z,      # 垂直最大速度限制
        max_yaw_rate=1.5,              # 最大航向角速度 (rad/s)
        # 下层 PID 参数 (匹配 PX4 默认值)
        # 使用默认值即可，已在 UAVVelocityWithDynamicsActionCfg 中设置
    )


@configclass
class ObservationsCfg:
    """Observation configuration."""
    
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity,)
        pose_command = ObsTerm(func=mdp.pose_command_position_2d, params={"command_name": "pose_command"})
        actions = ObsTerm(func=mdp.last_action)
        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner")},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    
    @configclass
    class PolicyCfgUAV(ObsGroup):
        """Observations for UAV policy group (3D navigation)."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=mdp.base_height)  # 高度观测 (1维)
        # base_ang_vel = ObsTerm(func=mdp.base_ang_vel)  # 质点模型，角速度始终为0
        # projected_gravity = ObsTerm(func=mdp.projected_gravity,)  # 质点模型，姿态不变，始终为[0,0,-1]
        pose_command = ObsTerm(func=mdp.pose_command_position_only, params={"command_name": "pose_command"})
        actions = ObsTerm(func=mdp.last_action)
        lidar_scan = ObsTerm(
            func=mdp.lidar_scan,
            params={"sensor_cfg": SceneEntityCfg("lidar_scanner")},
            clip=(-5.0, 5.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    policy_uav: PolicyCfgUAV = PolicyCfgUAV()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    """
    以下两种机器人重置方式二选一
    """
    # 1.在障碍物区域边界随机重置机器人
    # reset_base = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x": (-25.0, -25.0), "y": (-25.0, 25.0), "yaw": (0.0, 0.0)},
    #         "velocity_range": {
    #             "x": (0.0, 0.0),
    #             "y": (0.0, 0.0),
    #             "z": (0.0, 0.0),
    #             "roll": (0.0, 0.0),
    #             "pitch": (0.0, 0.0),
    #             "yaw": (0.0, 0.0),
    #         },
    #     },
    # )

    # 2.在障碍物区域内部随机重置机器人
    reset_base = EventTerm(
        func=mdp.reset_root_state_from_terrain,
        mode="reset",
        params={
            "pose_range": {"yaw": (-3.14159, 3.14159), "z": (1.0, 5.0)},  # yaw随机, 高度1-5m
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "offset_z": 0.0,  # 高度由pose_range控制，offset设为0
        },
    )


    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )




@configclass
class RewardsCfg:
    """Reward terms for the MDP - optimized for position-based navigation."""

    # 终止惩罚
    termination_penalty = RewTerm(
        # Only penalize "failure" terminations. Do not penalize success termination (reach_target),
        # otherwise the policy may learn to hover near the goal to avoid ending the episode.
        func=mdp.is_terminated_term,
        weight=-200.0,  # Reduced from -2000.0 to prevent gradient explosion
        params={"term_keys": ["out_of_bounds", "out_of_height_limit", "least_lidar_depth", "roll_over"]},
    )

    # 位置跟踪奖励 - 鼓励机器人接近目标点 (2D平面导航)
    # position_tracking_abs = RewTerm(
    #     func=mdp.position_command_error_2d,  # 使用2D位置奖励函数
    #     weight=0.1,
    #     params={"origin_distance": 50.0, "command_name": "pose_command"},
    # )
    
    # 位置跟踪奖励 - 鼓励机器人接近目标点 (3D空间导航)
    # 原来 weight=1 太小，相对于 termination_penalty=-200 无法驱动学习
    position_tracking_abs_3d = RewTerm(
        func=mdp.position_command_error_abs,  # 使用3D位置奖励函数
        weight=10.0,  # 增大权重，让位置奖励更显著
        params={"origin_distance": 20.0, "command_name": "pose_command"},  # 减小 origin_distance 让奖励更敏感
    )

    # 到达目标点奖励 - 大奖励鼓励完成任务
    reach_target_reward = RewTerm(
        func=mdp.reach_target_reward,
        weight=400.0,  # Reduced from 4000.0 to prevent gradient explosion
        params={"threshold": 2, "command_name": "pose_command"},  # 与终止条件保持一致
    )

    # 动作平滑性惩罚 - 鼓励平滑运动 (训练初期适当降低)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)  
    
    # 目标点附近速度惩罚 - 鼓励UAV减速停稳 (已注释掉)
    # velocity_near_target = RewTerm(
    #     func=mdp.velocity_near_target_penalty,
    #     weight=-2.0,  # 负权重，惩罚高速度
    #     params={"command_name": "pose_command", "distance_threshold": 5.0},
    # )  
    

    # 朝向奖励 - 鼓励机器人朝向目标
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_error_abs,
    #     weight=-0.5,  # 朝向误差惩罚
    #     params={"command_name": "pose_command"},
    # )
    
    #


    # flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-100.0)

    # 障碍物距离惩罚 - 鼓励机器人远离障碍物
    # lidar_depth_min = RewTerm(
    #     func=mdp.lidar_depth_min, 
    #     weight=-1000,  # 减少惩罚权重从-1000到-100
    #     params={"threshold": 0.12},
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)  # 超时时重置

    out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": -1.0},
    )

    out_of_height_limit = DoneTerm(
        func=mdp.out_of_height_limit,
        params={"asset_cfg": SceneEntityCfg("robot"), "min_height": 1.0, "max_height": 5.0},
    )

    least_lidar_depth = DoneTerm(
        func=mdp.least_lidar_depth,
        params={"sensor_cfg": SceneEntityCfg("lidar_scanner"), "threshold": 0.12},  # 放宽碰撞阈值 30cm
    )

    roll_over = DoneTerm(
        func=mdp.roll_over,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2},
    )

    # 接近目标点时终止任务并重置 (只需位置到达，不再要求速度减小)
    reach_target = DoneTerm(
        func=mdp.reach_target,
        params={"threshold": 2, "command_name": "pose_command"},
        # velocity_threshold 已移除，不再要求速度减小
    )

    


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class ExplorationVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the exploration environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=256, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settingsd
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [0, -40, 60]
        self.viewer.lookat = [1.5, 0, -5.5]
        
        # general settings
        self.decimation = 10 
        self.episode_length_s = 180.0
        # simulation settings
        self.sim.dt = 0.01  # 增大仿真时间步到0.01秒
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.lidar_scanner is not None:
            self.scene.lidar_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


##
# Configuration functions for different robot types
##

def create_ugv_config() -> ExplorationVelocityRoughEnvCfg:
    """Create configuration for UGV (ground vehicle)."""
    cfg = ExplorationVelocityRoughEnvCfg()
    
    # 禁用UAV动作，只使用UGV动作
    cfg.actions.uav_action = None
    
    # 禁用UAV观察，只使用UGV观察
    cfg.observations.policy_uav = None
    
    return cfg


def create_uav_config() -> ExplorationVelocityRoughEnvCfg:
    """Create configuration for UAV (aerial vehicle)."""
    cfg = ExplorationVelocityRoughEnvCfg()
    
    # 禁用UGV动作，只使用UAV动作
    cfg.actions.ugv_action = None
    
    # 禁用UGV观察，只使用UAV观察
    cfg.observations.policy = None
    
    return cfg


def create_config(robot_type: RobotType = DEFAULT_ROBOT_TYPE) -> ExplorationVelocityRoughEnvCfg:
    """Create configuration for specified robot type.
    
    Args:
        robot_type: The type of robot (UGV or UAV)
        
    Returns:
        Configuration for the specified robot type
    """
    if robot_type == RobotType.UGV:
        return create_ugv_config()
    elif robot_type == RobotType.UAV:
        return create_uav_config()
    else:
        raise ValueError(f"Unknown robot type: {robot_type}")


# 使用示例:
# 
# # 创建UGV配置
# ugv_cfg = create_ugv_config()
# 
# # 创建UAV配置  
# uav_cfg = create_uav_config()
# 
# # 或者直接指定机器人类型
# cfg = create_config(RobotType.UGV)  # UGV配置
# cfg = create_config(RobotType.UAV)  # UAV配置
