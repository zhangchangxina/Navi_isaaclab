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
        from isaaclab.sensors import patterns
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


        # actions
        self.actions.uav_action = None
        self.actions.ugv_action.body_name = "base_footprint"
        
        # observations - 禁用UAV观察，只使用UGV观察
        self.observations.policy_uav = None



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
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False

