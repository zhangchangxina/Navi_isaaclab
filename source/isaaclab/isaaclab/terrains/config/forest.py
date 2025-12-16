# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
import isaaclab.terrains.trimesh as mesh_gen


from ..terrain_generator_cfg import TerrainGeneratorCfg
from ..sub_terrain_cfg import FlatPatchSamplingCfg

PLANE_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(50.0, 50.0),
    border_width=15.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
    },
)


FOREST_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=0, # adjustable, used for save and load terrains
    size=(50.0, 50.0),
    border_width=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.1,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "hf_forest0": terrain_gen.MeshForestTerrainCfg(
            proportion=0.25, 
            obstacle_height_range=(2.0, 5.0), 
            obstacle_radius_range=(0.25, 1.0), 
            num_obstacles=250,
            flat_patch_sampling = {
                "init_pos": FlatPatchSamplingCfg(num_patches=1000, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
                "target": FlatPatchSamplingCfg(num_patches=1000, patch_radius=0.5, max_height_diff=0.05, z_range=(-100.0, 1.0)),
            }
        ),

    },
)
"""Forest terrains configuration."""
