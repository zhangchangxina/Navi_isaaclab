# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.exploration.velocity import mdp

from isaaclab_tasks.manager_based.exploration.velocity.velocity_env_cfg import ExplorationVelocityRoughEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.drone import DRONE_CFG  # isort: skip


@configclass
class DroneRoughEnvCfg(ExplorationVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = DRONE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.lidar_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Drone雷达配置 - 基于真实激光雷达参数
        from isaaclab.sensors import patterns
        self.scene.lidar_scanner.pattern_cfg = patterns.LidarPatternCfg(
            # 真实激光雷达配置（4线激光雷达）
            channels=8, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=10.0
            # channels=16, vertical_fov_range=(-7.0, 52.0), horizontal_fov_range=(-180.0, 180.0), horizontal_res=5.0
            # 低分辨率配置（训练用，与检查点兼容）
            # channels=1, vertical_fov_range=(0.0, 1.0), horizontal_fov_range=(-90.0, 90.0), horizontal_res=5.0
        )

        # self.commands.pose_command = mdp.TerrainBasedPoseCommandCfg(
        #     asset_name="robot",
        #     resampling_time_range=(180.0, 180.0),
        #     debug_vis=True,
        #     ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(pos_x=(25.5, 25.5), pos_y=(-25.0, 25.0),pos_z=(0.0, 4.0)),
        #     # ranges=mdp.TerrainBasedPoseCommandCfg.Ranges(pos_z=(0.0, 4.0)),
        #     offset_z = 0.2
        # )


        # self.commands.pose_command = mdp.UniformPose2dCommandCfg(
        #     asset_name="robot",
        #     simple_heading=False,
        #     resampling_time_range=(180.0, 180.0),
        #     debug_vis=True,
        #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(25.5, 25.5), pos_y=(-25.0, 25.0), heading=(1.57, 4.71)),
        # )

        self.events.add_base_mass = None

        self.actions.ugv_action = None
        self.actions.uav_action.scale = [0.2, 0.2, 0.02]
        self.actions.uav_action.body_name = "base"
        
        # observations - 使用UAV观察配置
        self.observations.policy = self.observations.policy_uav

        # terminations
        self.terminations.roll_over.params["threshold"] = 0.99



@configclass
class DroneRoughEnvCfg_PLAY(DroneRoughEnvCfg):
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
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

