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
from dataclasses import dataclass
from typing import Union

import gymnasium
import mujoco
import numpy as np
import safety_gymnasium
from gymnasium.envs.mujoco.mujoco_rendering import RenderContextOffscreen
from safety_gymnasium import agents
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.geoms import GEOMS_REGISTER
from safety_gymnasium.assets.mocaps import MOCAPS_REGISTER
from safety_gymnasium.assets.objects import OBJS_REGISTER
from safety_gymnasium.utils.common_utils import MujocoException
from safety_gymnasium.utils.keyboard_viewer import KeyboardViewer
from safety_gymnasium.utils.random_generator import RandomGenerator
from safety_gymnasium.world import World


@dataclass
class RenderConf:
    """Render options."""

    labels: bool = False
    lidar_markers: bool = True
    lidar_radius: float = 0.15
    lidar_size: float = 0.025
    lidar_offset_init: float = 0.5
    lidar_offset_delta: float = 0.06


@dataclass
class PlacementsConf:
    """Placement options."""

    placements = None  # this is generated during running
    extents = [-2, -2, 2, 2]  # Placement limits (min X, min Y, max X, max Y)
    margin = 0.0  # Additional margin added to keepout when placing objects


@dataclass
class SimulationConf:
    """Simulation options."""

    # Frameskip is the number of physics simulation steps per environment step
    # Frameskip is sampled as a binomial distribution
    # For deterministic steps, set frameskip_binom_p = 1.0 (always take max frameskip)
    # Number of draws trials in binomial distribution (max frameskip)
    frameskip_binom_n: int = 10
    # Probability of trial return (controls distribution)
    frameskip_binom_p: float = 1.0


@dataclass
class VisionEnvConf:
    """Vision observation parameters"""

    vision_size = (
        60,
        40,
    )  # Size (width, height) of vision observation;
    # gets flipped internally to (rows, cols) format


@dataclass
class FloorConf:
    """Floor options."""

    type: str = 'mat'  # choose from 'mat' and 'village'
    size: tuple = (3.5, 3.5, 0.1)  # Size of floor in environments


@dataclass
class WorldInfo:
    """World information."""

    layout: dict = None
    reset_layout: dict = None
    world_config_dict: dict = None


class Underlying(abc.ABC):  # pylint: disable=too-many-instance-attributes
    """Base class which is in charge of mujoco and underlying process.

    In short: The engine for the Safety Gymnasium environment.
    """

    def __init__(self, config=None):
        """Initialize the engine."""

        self.sim_conf = SimulationConf()
        self.placements_conf = PlacementsConf()
        self.render_conf = RenderConf()
        self.vision_env_conf = VisionEnvConf()
        self.floor_conf = FloorConf()

        self.random_generator = RandomGenerator()

        self.world = None
        self.world_info = WorldInfo()

        self.viewer = None
        self._viewers = {}

        # Obstacles which are added in environments.
        self._geoms = {}
        self._objects = {}
        self._mocaps = {}

        # something are parsed from pre-defined configs
        self.agent_name = None
        self.observe_vision = False  # Observe vision from the agent
        self.debug = False
        self._parse(config)
        self.observation_flatten = True  # Flatten observation into a vector
        self.agent = None
        self.action_noise: float = (
            0.0  # Magnitude of independent per-component gaussian action noise
        )
        self._build_agent(self.agent_name)

    def _parse(self, config):
        """Parse a config dict.

        Modify some attributes according to config.
        So that easily adapt to different environment settings.
        """
        for key, value in config.items():
            if '.' in key:
                obj, key = key.split('.')
                assert hasattr(self, obj) and hasattr(getattr(self, obj), key), f'Bad key {key}'
                setattr(getattr(self, obj), key, value)
            else:
                assert hasattr(self, key), f'Bad key {key}'
                setattr(self, key, value)

    def _build_agent(self, agent_name):
        """Build the agent in the world."""
        assert hasattr(agents, agent_name), 'agent not found'
        agent_cls = getattr(agents, agent_name)
        self.agent = agent_cls(random_generator=self.random_generator)

    def _add_geoms(self, *geoms):
        """Add geom type objects into environments and set corresponding attributes."""
        for geom in geoms:
            assert (
                type(geom) in GEOMS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._geoms[geom.name] = geom
            setattr(self, geom.name, geom)
            geom.set_agent(self.agent)

    def _add_objects(self, *objects):
        """Add object type objects into environments and set corresponding attributes."""
        for obj in objects:
            assert (
                type(obj) in OBJS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._objects[obj.name] = obj
            setattr(self, obj.name, obj)
            obj.set_agent(self.agent)

    def _add_mocaps(self, *mocaps):
        """Add mocap type objects into environments and set corresponding attributes."""
        for mocap in mocaps:
            assert (
                type(mocap) in MOCAPS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._mocaps[mocap.name] = mocap
            setattr(self, mocap.name, mocap)
            mocap.set_agent(self.agent)

    def reset(self):
        """Reset the environment."""
        self._build()
        # Save the layout at reset
        self.world_info.reset_layout = deepcopy(self.world_info.layout)

    def _build(self):
        """Build a new physics simulation environment"""
        if self.placements_conf.placements is None:
            self._build_placements_dict()
            self.random_generator.set_placements_info(
                self.placements_conf.placements,
                self.placements_conf.extents,
                self.placements_conf.margin,
            )
        # Sample object positions
        self.world_info.layout = self.random_generator.build_layout()

        # Build the underlying physics world
        self.world_info.world_config_dict = self._build_world_config(self.world_info.layout)

        if self.world is None:
            self.world = World(self.agent, self._obstacles, self.world_info.world_config_dict)
            self.world.reset()
            self.world.build()
        else:
            self.world.reset(build=False)
            self.world.rebuild(self.world_info.world_config_dict, state=False)
            if self.viewer:
                self._update_viewer(self.model, self.data)

    def simulation_forward(self, action):
        """Take a step in the physics simulation."""
        # Simulate physics forward
        if self.debug:
            self.agent.debug()
        else:
            noise = (
                self.action_noise * self.random_generator.randn(self.agent.body_info.nu)
                if self.action_noise
                else None
            )
            self.agent.apply_action(action, noise)

        exception = False
        for _ in range(
            self.random_generator.binomial(
                self.sim_conf.frameskip_binom_n, self.sim_conf.frameskip_binom_p
            )
        ):
            try:
                for mocap in self._mocaps.values():
                    mocap.move()
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

    def update_layout(self):
        """Update layout dictionary with new places of objects."""
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member
        for k in list(self.world_info.layout.keys()):
            # Mocap objects have to be handled separately
            if 'gremlin' in k:
                continue
            self.world_info.layout[k] = self.data.body(k).xpos[:2].copy()

    def _set_goal(self, pos):
        """Set the goal position."""
        if pos.shape == (2,):
            self.model.body('goal').pos[:2] = pos[:2]
        elif pos.shape == (3,):
            self.model.body('goal').pos[:2] = pos[:2]
        else:
            raise NotImplementedError

    def _render_lidar(self, poses, color, offset, group):
        """Render the lidar observation."""
        agent_pos = self.agent.pos
        agent_mat = self.agent.mat
        lidar = self._obs_lidar(poses, group)
        for i, sensor in enumerate(lidar):
            if self.lidar_conf.type == 'pseudo':  # pylint: disable=no-member
                i += 0.5  # Offset to center of bin
            theta = 2 * np.pi * i / self.lidar_conf.num_bins  # pylint: disable=no-member
            rad = self.render_conf.lidar_radius
            binpos = np.array([np.cos(theta) * rad, np.sin(theta) * rad, offset])
            pos = agent_pos + np.matmul(binpos, agent_mat.transpose())
            alpha = min(1, sensor + 0.1)
            self.viewer.add_marker(
                pos=pos,
                size=self.render_conf.lidar_size * np.ones(3),
                type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
                rgba=np.array(color) * alpha,
                label='',
            )

    def _render_compass(self, pose, color, offset):
        """Render a compass observation."""
        agent_pos = self.agent.pos
        agent_mat = self.agent.mat
        # Truncate the compass to only visualize XY component
        compass = np.concatenate([self._obs_compass(pose)[:2] * 0.15, [offset]])
        pos = agent_pos + np.matmul(compass, agent_mat.transpose())
        self.viewer.add_marker(
            pos=pos,
            size=0.05 * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * 0.5,
            label='',
        )

    # pylint: disable-next=too-many-arguments
    def _render_area(self, pos, size, color, label='', alpha=0.1):
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
            label=label if self.render_conf.labels else '',
        )

    # pylint: disable-next=too-many-arguments
    def _render_sphere(self, pos, size, color, label='', alpha=0.1):
        """Render a radial area in the environment."""
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.r_[pos, 0]  # Z coordinate 0
        self.viewer.add_marker(
            pos=pos,
            size=size * np.ones(3),
            type=mujoco.mjtGeom.mjGEOM_SPHERE,  # pylint: disable=no-member
            rgba=np.array(color) * alpha,
            label=label if self.render_conf.labels else '',
        )

    # pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements
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

        self._get_viewer(mode)

        # Turn all the geom groups on
        self.viewer.vopt.geomgroup[:] = 1

        # Lidar and Compass markers
        if self.render_conf.lidar_markers:
            offset = (
                self.render_conf.lidar_offset_init
            )  # Height offset for successive lidar indicators
            for obstacle in self._obstacles:
                if obstacle.is_lidar_observed:
                    self._render_lidar(obstacle.pos, obstacle.color, offset, obstacle.group)
                if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                    self._render_compass(
                        getattr(self, obstacle.name + '_pos'), obstacle.color, offset
                    )
                offset += self.render_conf.lidar_offset_delta

        # Add indicator for nonzero cost
        if cost.get('cost', 0) > 0:
            self._render_sphere(self.agent.pos, 0.25, COLOR['red'], alpha=0.5)

        # Draw vision pixels
        if mode == 'rgb_array':
            self._get_viewer(mode).render(camera_id=camera_id)
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
        'safety_gymnasium.utils.keyboard_viewer.KeyboardViewer',
        'gymnasium.envs.mujoco.mujoco_rendering.RenderContextOffscreen',
    ]:
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = KeyboardViewer(
                    self.model, self.data, self.agent.keyboard_control_callback
                )
            elif mode in {'rgb_array', 'depth_array'}:
                self.viewer = RenderContextOffscreen(self.model, self.data)
            else:
                raise AttributeError(f'Unexpected mode: {mode}')

            # self.viewer_setup()
            self._viewers[mode] = self.viewer

        return self.viewer

    def _update_viewer(self, model, data):
        """update the viewer with new model and data"""
        assert self.viewer, 'Call before self.viewer existing.'
        self.viewer.model = model
        self.viewer.data = data

    @abc.abstractmethod
    def _obs_lidar(self, positions, group):
        """Calculate and return a lidar observation.  See sub methods for implementation."""

    @abc.abstractmethod
    def _obs_compass(self, pos):
        """Return a agent-centric compass observation of a list of positions.

        Compass is a normalized (unit-length) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """

    @abc.abstractmethod
    def _build_placements_dict(self):
        """Build a dict of placements.  Happens only once."""

    @abc.abstractmethod
    def _build_world_config(self, layout):
        """Create a world_config from our own config."""

    @property
    def model(self):
        """Helper to get the world's model instance."""
        return self.world.model

    @property
    def data(self):
        """Helper to get the world's simulation data instance."""
        return self.world.data

    @property
    def _obstacles(self):
        """Get the obstacles in the task."""
        return (
            list(self._geoms.values()) + list(self._objects.values()) + list(self._mocaps.values())
        )
