# Copyright 2022 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Base mujoco task."""

import abc
from copy import deepcopy
from typing import Union

import gymnasium  # pylint: disable=unused-import
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import RenderContextOffscreen, Viewer
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.utils.common_utils import MujocoException, ResamplingError
from safety_gymnasium.world import World


class BaseMujocoTask(
    abc.ABC
):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Base class which is in charge of mujoco and underlying process.

    In short: The engine for the Safety Gymnasium environment.
    """

    def __init__(self, config=None):
        """Initialize the engine."""
        # Default configuration

        # Render options
        self.render_labels = False
        self.render_lidar_markers = True
        self.render_lidar_radius = 0.15
        self.render_lidar_size = 0.025
        self.render_lidar_offset_init = 0.5
        self.render_lidar_offset_delta = 0.06

        # Frameskip is the number of physics simulation steps per environment step
        # Frameskip is sampled as a binomial distribution
        # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
        # Number of draws trials in binomial distribution (max frameskip)
        self.frameskip_binom_n = 10
        # Probability of trial return (controls distribution)
        self.frameskip_binom_p = 1.0

        # Vision observation parameters
        self.vision_size = (
            60,
            40,
        )  # Size (width, height) of vision observation;
        self.parse(config)

        self.random_generator = None

        self.placements = None
        self.placements_extents = None
        self.placements_margin = None
        self.layout = None
        self.reset_layout = None
        self.world = None
        self.world_config_dict = None
        self.last_action = None
        self.last_subtreecom = None

        self.action_space = None
        self.action_noise = None
        self.observation_space = None
        self.obs_space_dict = None
        self.lidar_type = None
        self.lidar_num_bins = None

        self.viewer = None
        self._viewers = {}
        self._geoms = None
        self._objects = None
        self._mocaps = None

    def parse(self, config):
        """Parse a config dict.

        Modify some attributes according to config.
        So that easily adapt to different environment settings.
        """
        for key, value in config.items():
            setattr(self, key, value)

    def clear(self):
        """Reset internal state for building."""
        self.layout = None

    def reset(self):
        """Reset the environment."""
        self.clear()
        self.build()
        # Save the layout at reset
        self.reset_layout = deepcopy(self.layout)

    def build(self):
        """Build a new physics simulation environment"""
        # Sample object positions
        self.build_layout()

        # Build the underlying physics world
        self.world_config_dict = self.build_world_config(self.layout)

        if self.world is None:
            self.world = World(self.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_config_dict, state=False)
            if self.viewer:
                self.update_viewer(self.model, self.data)

        # Save last action
        self.last_action = np.zeros(self.action_space.shape)

        # Save last subtree center of mass
        self.last_subtreecom = self.world.get_sensor('subtreecom')

    def build_layout(self):
        """Rejection sample a placement of objects to find a layout."""
        for _ in range(10000):
            if self.sample_layout():
                break
        else:
            raise ResamplingError('Failed to sample layout of objects')

    def update_layout(self):
        """Update layout dictionary with new places of objects."""
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member
        for k in list(self.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.layout[k] = self.data.body(k).xpos[:2].copy()

    def apply_action(self, action):
        """Apply an action to the robot."""
        action = np.array(action, copy=False)  # Cast to ndarray

        # Set action
        action_range = self.model.actuator_ctrlrange

        self.data.ctrl[:] = np.clip(
            action, action_range[:, 0], action_range[:, 1]
        )  # np.clip(action * 2 / action_scale, -1, 1)
        if self.action_noise:
            self.data.ctrl[:] += self.action_noise * self.random_generator.randn(self.model.nu)

        # Simulate physics forward
        exception = False
        for _ in range(
            self.random_generator.binomial(self.frameskip_binom_n, self.frameskip_binom_p)
        ):
            try:
                self.set_mocaps()
                # pylint: disable-next=no-member
                mujoco.mj_step(self.model, self.data)  # Physics simulation step
            except MujocoException as me:  # pylint: disable=invalid-name
                print('MujocoException', me)
                exception = True
                break
        if exception:
            return exception

        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor readings correct!
        return exception

    def set_random_generator(self, random_generator):
        """Set the random state of the engine."""
        self.random_generator = random_generator

    def set_mocaps(self):
        """Set mocap object positions before a physics step is executed."""
        if len(self._mocaps):
            raise NotImplementedError(
                'Please Implement your specific set_mocaps() function for mocaps in your task.'
            )

    def sample_layout(self):
        """Sample a single layout, returning True if successful, else False."""
        if self.placements is None:
            self.build_placements_dict()

        def placement_is_valid(xy, layout):  # pylint: disable=invalid-name
            for other_name, other_xy in layout.items():
                other_keepout = self.placements[other_name][1]
                dist = np.sqrt(np.sum(np.square(xy - other_xy)))
                if dist < other_keepout + self.placements_margin + keepout:
                    return False
            return True

        layout = {}
        for name, (placements, keepout) in self.placements.items():
            conflicted = True
            for _ in range(100):
                # pylint: disable-next=invalid-name
                xy = self.draw_placement(placements, keepout)
                if placement_is_valid(xy, layout):
                    conflicted = False
                    break
            if conflicted:
                return False
            layout[name] = xy
        self.layout = layout
        return True

    def sample_goal_position(self):
        """Sample a new goal position and return True, else False if sample rejected."""
        placements, keepout = self.placements['goal']
        goal_xy = self.draw_placement(placements, keepout)
        for other_name, other_xy in self.layout.items():
            other_keepout = self.placements[other_name][1]
            dist = np.sqrt(np.sum(np.square(goal_xy - other_xy)))
            if dist < other_keepout + self.placements_margin + keepout:
                return False
        self.layout['goal'] = goal_xy
        return True

    def draw_placement(self, placements, keepout):
        """Sample an (x,y) location, based on potential placement areas.

        Summary of behavior:

        'placements' is a list of (xmin, xmax, ymin, ymax) tuples that specify
        rectangles in the XY-plane where an object could be placed.

        'keepout' describes how much space an object is required to have
        around it, where that keepout space overlaps with the placement rectangle.

        To sample an (x,y) pair, first randomly select which placement rectangle
        to sample from, where the probability of a rectangle is weighted by its
        area. If the rectangles are disjoint, there's an equal chance the (x,y)
        location will wind up anywhere in the placement space. If they overlap, then
        overlap areas are double-counted and will have higher density. This allows
        the user some flexibility in building placement distributions. Finally,
        randomly draw a uniform point within the selected rectangle.
        """
        if placements is None:
            choice = self.constrain_placement(self.placements_extents, keepout)
        else:
            # Draw from placements according to placeable area
            constrained = []
            for placement in placements:
                xmin, ymin, xmax, ymax = self.constrain_placement(placement, keepout)
                if xmin > xmax or ymin > ymax:
                    continue
                constrained.append((xmin, ymin, xmax, ymax))
            assert constrained, 'Failed to find any placements with satisfy keepout'
            if len(constrained) == 1:
                choice = constrained[0]
            else:
                areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in constrained]
                probs = np.array(areas) / np.sum(areas)
                choice = constrained[self.random_generator.choice(len(constrained), p=probs)]
        xmin, ymin, xmax, ymax = choice
        return np.array(
            [self.random_generator.uniform(xmin, xmax), self.random_generator.uniform(ymin, ymax)]
        )

    def constrain_placement(self, placement, keepout):
        """Helper function to constrain a single placement by the keepout radius."""
        xmin, ymin, xmax, ymax = placement
        return (xmin + keepout, ymin + keepout, xmax - keepout, ymax - keepout)

    def render_lidar(self, poses, color, offset, group):
        """Render the lidar observation."""
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        lidar = self.obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.lidar_type == 'pseudo':
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.lidar_num_bins
            rad = self.render_lidar_radius
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = robot_pos + np.matmul(binpos, robot_mat.transpose())
            alpha = min(1, sensor + 0.1)
            self.viewer.add_marker(
                pos=pos,
                size=self.render_lidar_size * np.ones(3),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
                rgba=np.array(color) * alpha,
                label='',
            )

    def render_compass(self, pose, color, offset):
        """Render a compass observation."""
        robot_pos = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        # Truncate the compass to only visualize XY component
        compass = np.concatenate([self.obs_compass(pose)[:2] * 0.15, [offset]])
        pos = robot_pos + np.matmul(compass, robot_mat.transpose())
        self.viewer.add_marker(
            pos=pos,
            size=0.05 * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * 0.5,
            label='',
        )

    # pylint: disable-next=too-many-arguments
    def render_area(self, pos, size, color, label='', alpha=0.1):
        """Render a radial area in the environment."""
        z_size = min(size, 0.3)
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=[size, size, z_size],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_labels else '',
        )

    # pylint: disable-next=too-many-arguments
    def render_sphere(self, pos, size, color, label='', alpha=0.1):
        """Render a radial area in the environment."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=size * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_labels else '',
        )

    # pylint: disable-next=too-many-arguments,too-many-branches
    def render(self, width, height, mode, camera_id=None, camera_name=None, cost=None):
        """Render the environment to the screen."""
        self.model.vis.global_.offwidth = width
        self.model.vis.global_.offheight = height

        if mode in {
            'rgb_array',
            'depth_array',
        }:

            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    'Both `camera_id` and `camera_name` cannot be' + ' specified at the same time.'
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = 'vision'

            if camera_id is None:
                # pylint: disable-next=no-member
                camera_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_CAMERA,  # pylint: disable=no-member
                    camera_name,
                )

                self._get_viewer(mode).render(camera_id=camera_id)

        if mode == 'human':
            self._get_viewer(mode)
            # pylint: disable-next=no-member
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

        # Turn all the geom groups on
        self.viewer.vopt.geomgroup[:] = 1

        # Lidar markers
        if self.render_lidar_markers:
            offset = self.render_lidar_offset_init  # Height offset for successive lidar indicators
            for geom in self._geoms.values():
                if geom.is_observe_lidar:
                    self.render_lidar(
                        getattr(self, geom.name + '_pos'), geom.color, offset, geom.group
                    )
                if hasattr(geom, 'is_observe_comp') and geom.is_observe_comp:
                    self.render_compass(getattr(self, geom.name + '_pos'), geom.color, offset)
                offset += self.render_lidar_offset_delta
            for obj in self._objects.values():
                if obj.is_observe_lidar:
                    self.render_lidar(
                        getattr(self, obj.name + '_pos'), obj.color, offset, obj.group
                    )
                if hasattr(obj, 'is_observe_comp') and obj.is_observe_comp:
                    self.render_compass(getattr(self, obj.name + '_pos'), obj.color, offset)
                offset += self.render_lidar_offset_delta
            for mocap in self._mocaps.values():
                if mocap.is_observe_lidar:
                    self.render_lidar(
                        getattr(self, mocap.name + '_pos'), mocap.color, offset, mocap.group
                    )
                if hasattr(mocap, 'is_observe_comp') and mocap.is_observe_comp:
                    self.render_compass(getattr(self, mocap.name + '_pos'), mocap.color, offset)
                offset += self.render_lidar_offset_delta

        # Add goal marker
        if 'buttons' in self._geoms:
            self.render_area(
                getattr(self, 'goal' + '_pos'),
                self.buttons.size * 2,  # pylint: disable=no-member
                COLOR['button'],
                'goal',
                alpha=0.1,
            )
            self.render_lidar(getattr(self, 'goal' + '_pos'), COLOR['goal'], offset, GROUP['goal'])

        # Add indicator for nonzero cost
        if cost.get('cost', 0) > 0:
            self.render_sphere(self.world.robot_pos(), 0.25, COLOR['red'], alpha=0.5)

        # Draw vision pixels
        if mode == 'rgb_array':
            data = self._get_viewer(mode).read_pixels(depth=False)
            # original image is upside-down, so flip it
            self.viewer._markers[:] = []  # pylint: disable=protected-access
            self.viewer._overlays.clear()  # pylint: disable=protected-access
            return data[::-1, :, :]
        if mode == 'depth_array':
            self._get_viewer(mode).render()
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(depth=True)[1]
            # original image is upside-down, so flip it
            self.viewer._markers[:] = []  # pylint: disable=protected-access
            self.viewer._overlays.clear()  # pylint: disable=protected-access
            return data[::-1, :]
        if mode == 'human':
            self._get_viewer(mode).render()
            return None
        raise NotImplementedError(f'Render mode {mode} is not implemented.')

    def _get_viewer(
        self, mode
    ) -> Union[
        'gymnasium.envs.mujoco.mujoco_rendering.Viewer',
        'gymnasium.envs.mujoco.mujoco_rendering.RenderContextOffscreen',
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = Viewer(self.model, self.data)
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(f'Unexpected mode: {mode}')

            # self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def update_viewer(self, model, data):
        """update the viewer with new model and data"""
        assert self.viewer, 'Call before self.viewer existing.'
        self.viewer.model = model
        self.viewer.data = data

    @abc.abstractmethod
    def obs_lidar(self, positions, group):
        """Calculate and return a lidar observation.  See sub methods for implementation."""

    @abc.abstractmethod
    def obs_compass(self, pos):
        """Return a robot-centric compass observation of a list of positions.

        Compass is a normalized (unit-length) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

    @abc.abstractmethod
    def build_placements_dict(self):
        """Build a dict of placements.  Happens only once."""

    @abc.abstractmethod
    def build_world_config(self, layout):
        """Create a world_config from our own config."""

    @property
    def model(self):
        """Helper to get the world's model instance."""
        return self.world.model

    @property
    def data(self):
        """Helper to get the world's simulation data instance."""
        return self.world.data
