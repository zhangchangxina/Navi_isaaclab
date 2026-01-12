# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab_tasks.manager_based.exploration.velocity import mdp

from isaaclab_tasks.manager_based.exploration.velocity.velocity_env_cfg import ExplorationVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.turtlebot import TURTLEBOT_CFG  # isort: skip


@configclass
class TurtlebotRoughEnvCfg(ExplorationVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        self.scene.robot = TURTLEBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.lidar_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_footprint/base_link/base_scan"
        
        # Disable UAV observations for Turtlebot (only use policy group)
        self.observations.policy_uav = None

        # Turtlebot雷达配置 - 适合地面导航
        from isaaclab.sensors import RayCasterCfg, patterns
        # UGV 雷达水平安装，无前倾
        self.scene.lidar_scanner.offset = RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),   # 雷达在 base_scan 坐标系原点
            rot=(1.0, 0, 0, 0),   # 无旋转（水平）
        )
        self.scene.lidar_scanner.pattern_cfg = patterns.LidarPatternCfg(
            # 低分辨率配置（与训练时一致，用于加载检查点）
            channels=1, vertical_fov_range=(0.0, 1.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
            # 高分辨率配置（调试时使用）
            # channels=8, vertical_fov_range=(-10.0, 10.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=2.0
        )

        # self.commands.pose_command = mdp.TerrainBasedPoseCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(180.0, 180.0),
        #     debug_vis=True,
        #     ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(pos_z=(0.0, 0.0)),
        #     offset_z = 0.3
        # )

        self.events.add_base_mass.params["mass_distribution_params"] = (1.0, 1.0)
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_footprint"


        # actions - 禁用UAV动作，只使用UGV动作
        self.actions.uav_action = None

        # Turtlebot 仿真加速参数（约2倍真实速度，加速训练）
        # 训练完成后可降低速度 fine-tune
        self.actions.ugv_action.body_name = "base_footprint"
        self.actions.ugv_action.lin_scale = 0.5      # action=1 → 0.5 m/s (仿真加速)
        self.actions.ugv_action.ang_scale = 2.0      # action=1 → 2.0 rad/s (略低于真实)
        self.actions.ugv_action.max_lin_vel = 0.5    # 最大线速度限制
        self.actions.ugv_action.max_ang_vel = 2.0    # 最大角速度限制
        self.actions.ugv_action.lin_acc = 1.0        # 线加速度限制 (m/s²)
        self.actions.ugv_action.ang_acc = 2.0        # 角加速度限制 (rad/s²)
        
        # observations - 禁用UAV观察，只使用UGV观察
        self.observations.policy_uav = None

        # commands - UGV目标点稍微抬高便于可视化
        self.commands.pose_command.offset_z = 0.2

        # rewards - UGV使用2D距离奖励（地面导航，不考虑高度）
        self.rewards.position_tracking_abs_3d = RewTerm(
            func=mdp.position_command_error_2d,  # 使用2D位置奖励函数
            weight=1,
            params={"origin_distance": 50.0, "command_name": "pose_command"},
        )



@configclass
class TurtlebotRoughEnvCfg_PLAY(TurtlebotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

