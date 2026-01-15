from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TrajectoryVisCommandCfg

try:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
    DEBUG_DRAW_AVAILABLE = True
except ImportError:
    DEBUG_DRAW_AVAILABLE = False
    omni_debug_draw = None
import matplotlib.pyplot as plt

class TrajectoryVisCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: TrajectoryVisCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: TrajectoryVisCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        self.trajectory_points = None
        self.trajectory_vels = None

        self.cmap = plt.get_cmap('jet')  # 'plasma', 'magma', 'cividis', 'jet'
        self.vel_upper = 1.0  # 与 _UAV_MAX_VEL_HOR 一致

        self.traj_thickness = 3.0



    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self):
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.trajectory_points

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data


    def _resample_command(self, env_ids: Sequence[int]):
        pos_w = self.robot.data.root_pos_w
        linvel_w = self.robot.data.root_lin_vel_w

        if self.trajectory_points is None:
            self.trajectory_points = pos_w.clone().unsqueeze(1).repeat(1, self.cfg.max_length, 1)
        else:
            # self.trajectory_points = self.trajectory_points.roll(shifts=-1, dims=1)  # 沿dim=1左移1位
            self.trajectory_points[:, :-1] = self.trajectory_points[:, 1:]
            self.trajectory_points[:, -1, :] = pos_w

        if self.trajectory_vels is None:
            self.trajectory_vels = linvel_w.clone().unsqueeze(1).repeat(1, self.cfg.max_length, 1)
        else:
            # self.trajectory_vels = self.trajectory_vels.roll(shifts=-1, dims=1)  # 沿dim=1左移1位
            self.trajectory_vels[:, :-1] = self.trajectory_vels[:, 1:]
            self.trajectory_vels[:, -1, :] = linvel_w

        if not DEBUG_DRAW_AVAILABLE:
            return
            
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()


        adjacent_dis = torch.norm(self.trajectory_points[:, :-1, :] - self.trajectory_points[:, 1:, :], dim=-1)
        indices = torch.arange(adjacent_dis.shape[1], device=self.device).unsqueeze(0).repeat(adjacent_dis.shape[0], 1)
        filtered_indices = torch.where(adjacent_dis>self.cfg.threshold, indices, -1)
        max_indices = torch.max(filtered_indices, dim=1).values


        draw_interface.clear_lines()

        for i in range(self.trajectory_points.shape[0]):
            source_pos = self.trajectory_points[i, max_indices[i]+1:-1, :]
            target_pos = self.trajectory_points[i, max_indices[i]+2:, :]

            cur_vel = torch.norm(self.trajectory_vels[i, max_indices[i]+1:-1, :], dim=-1)
            # try:
            #     print(cur_vel.max())
            # except:
            #     pass
            colors = self.cmap(cur_vel.cpu().numpy() / self.vel_upper)

            # lines_colors = [[1.0, 0.0, 0.0, 0.8]] * source_pos.shape[0]
            line_thicknesses = [self.traj_thickness] * source_pos.shape[0]

            draw_interface.draw_lines(source_pos.tolist(), target_pos.tolist(), colors, line_thicknesses)
            # draw_interface.draw_points(source_pos.tolist(), colors, line_thicknesses)




        # # sample velocity commands
        # r = torch.empty(len(env_ids), device=self.device)
        # # -- linear velocity - x direction
        # self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        # # -- linear velocity - y direction
        # self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # # -- ang vel yaw - rotation around z
        # self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # # heading target
        # if self.cfg.heading_command:
        #     self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        #     # update heading envs
        #     self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        # # update standing envs
        # self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        # if self.cfg.heading_command:
        #     # resolve indices of heading envs
        #     env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        #     # compute angular velocity
        #     heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
        #     self.vel_command_b[env_ids, 2] = torch.clip(
        #         self.cfg.heading_control_stiffness * heading_error,
        #         min=self.cfg.ranges.ang_vel_z[0],
        #         max=self.cfg.ranges.ang_vel_z[1],
        #     )
        # # Enforce standing (i.e., zero velocity command) for standing envs
        # # TODO: check if conversion is needed
        # standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        # self.vel_command_b[standing_env_ids, :] = 0.0
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
