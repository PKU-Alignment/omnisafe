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
"""Base task."""

import abc
import os
from collections import OrderedDict
from dataclasses import dataclass

import gymnasium
import mujoco
import numpy as np
import safety_gymnasium
import yaml
from safety_gymnasium.bases.underlying import Underlying
from safety_gymnasium.utils.common_utils import ResamplingError
from safety_gymnasium.utils.task_utils import theta2vec


@dataclass
class LidarConf:
    """Lidar options.

    Lidar observation parameters.
    """

    num_bins: int = 16  # Bins (around a full circle) for lidar sensing
    max_dist: float = 3  # Maximum distance for lidar sensitivity (if None, exponential distance)
    exp_gain: float = 1.0  # Scaling factor for distance in exponential distance lidar
    type: str = 'pseudo'  # 'pseudo', 'natural', see self._obs_lidar()
    alias: bool = True  # Lidar bins alias into each other


@dataclass
class CompassConf:
    """Compass options.

    Compass observation parameters.
    """

    shape = 2  # Set to 2 or 3 for XY or XYZ unit vector compass observation.


@dataclass
class RewardConf:
    """Reward options."""

    reward_orientation = False  # Reward for being upright
    reward_orientation_scale = 0.002  # Scale for uprightness reward
    reward_orientation_body = 'agent'  # What body to get orientation from
    reward_exception = -10.0  # Reward when encountering a mujoco exception
    reward_clip = 10  # Clip reward, last resort against physics errors causing magnitude spikes


@dataclass
class CostConf:
    """Cost options."""

    constrain_indicator = True  # If true, all costs are either 1 or 0 for a given step.


@dataclass
class MechanismConf:
    """Rule options."""

    # Starting position distribution
    randomize_layout = True  # If false, set the random seed before layout to constant
    continue_goal = True  # If true, draw a new goal after achievement
    terminate_resample_failure = True  # If true, end episode when resampling fails,
    # otherwise, raise a python exception.


@dataclass
class ObservationInfo:
    """Observation information."""

    obs_space_dict = None
    obs_flat_size = None


class BaseTask(Underlying):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Base task class for defining some common characteristic."""

    def __init__(self, config: dict):  # pylint: disable-next=too-many-statements
        super().__init__(config=config)

        self.num_steps = 1000  # Maximum number of environment steps in an episode

        self.lidar_conf = LidarConf()
        self.compass_conf = CompassConf()
        self.reward_conf = RewardConf()
        self.cost_conf = CostConf()
        self.mechanism_conf = MechanismConf()

        self.action_space = self.agent.action_space
        self.observation_space = None
        self.obs_info = ObservationInfo()

        self._is_load_static_geoms = False  # Whether to load static geoms in current task.

    def dist_goal(self):
        """Return the distance from the agent to the goal XY position."""
        return self.agent.dist_xy(self.goal_pos)

    def calculate_cost(self):
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Calculate constraint violations
        for obstacle in self._obstacles:
            cost.update(obstacle.cal_cost())

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        return cost

    def build_observation_space(self):
        """Construct observation space.  Happens only once at during __init__ in Builder."""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.agent.build_sensor_observation_space())

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                name = obstacle.name + '_' + 'lidar'
                obs_space_dict[name] = gymnasium.spaces.Box(
                    0.0, 1.0, (self.lidar_conf.num_bins,), dtype=np.float64
                )
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                gymnasium.spaces.Box(-1.0, 1.0, (self.compass_conf.shape,), dtype=np.float64)

        if self.observe_vision:
            width, height = self.vision_env_conf.vision_size
            rows, cols = height, width
            self.vision_env_conf.vision_size = (rows, cols)
            obs_space_dict['vision'] = gymnasium.spaces.Box(
                0, 255, self.vision_env_conf.vision_size + (3,), dtype=np.uint8
            )

        # Flatten it ourselves
        self.obs_info.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_info.obs_flat_size = sum(
                np.prod(i.shape) for i in self.obs_info.obs_space_dict.values()
            )
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_info.obs_flat_size,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def _build_placements_dict(self):
        """Build a dict of placements.

        Happens only once.
        """
        placements = {}

        placements.update(self._placements_dict_from_object('agent'))
        for obstacle in self._obstacles:
            placements.update(self._placements_dict_from_object(obstacle.name))

        self.placements_conf.placements = placements

    def toggle_observation_space(self):
        """Toggle observation space."""
        self.observation_flatten = not self.observation_flatten
        self.build_observation_space()

    def _build_world_config(self, layout):  # pylint: disable=too-many-branches
        """Create a world_config from our own config."""
        world_config = {}

        world_config['floor_type'] = self.floor_conf.type
        world_config['floor_size'] = self.floor_conf.size

        world_config['agent_base'] = self.agent.base
        world_config['agent_xy'] = layout['agent']
        if self.agent.rot is None:
            world_config['agent_rot'] = self.random_generator.random_rot()
        else:
            world_config['agent_rot'] = float(self.agent.rot)

        # process world config via different objects.
        world_config.update(
            {
                'geoms': {},
                'objects': {},
                'mocaps': {},
            }
        )
        for obstacle in self._obstacles:
            num = obstacle.num if hasattr(obstacle, 'num') else 1
            obstacle.process_config(world_config, layout, self.random_generator.generate_rots(num))
        if self._is_load_static_geoms:
            self._build_static_geoms_config(world_config['geoms'])

        return world_config

    def _build_static_geoms_config(self, geoms_config):
        """Load static geoms from .yaml file.

        Static geoms are geoms which won't be considered when calculate reward and cost.
        """
        env_info = self.__class__.__name__.split('Level')
        config_name = env_info[0].lower()
        level = int(env_info[1])

        # load all config of meshes in specific environment from .yaml file
        base_dir = os.path.dirname(safety_gymnasium.__file__)
        with open(
            os.path.join(base_dir, f'configs/{config_name}.yaml'), 'r', encoding='utf-8'
        ) as file:
            meshes_config = yaml.load(file, Loader=yaml.FullLoader)

        for idx in range(level + 1):
            for group in meshes_config[idx].values():
                geoms_config.update(group)

    def build_goal_position(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # Resample until goal is compatible with layout
        if 'goal' in self.world_info.layout:
            del self.world_info.layout['goal']
        for _ in range(10000):  # Retries
            if self.random_generator.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_info.world_config_dict['geoms']['goal']['pos'][:2] = self.world_info.layout[
            'goal'
        ]
        self._set_goal(self.world_info.layout['goal'])
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def _placements_dict_from_object(self, object_name):
        """Get the placements dict subset just for a given object name."""
        placements_dict = {}

        assert hasattr(self, object_name), f'object{object_name} does not exist, but you use it!'
        data_obj = getattr(self, object_name)

        if hasattr(data_obj, 'num'):  # Objects with multiplicity
            object_fmt = object_name[:-1] + '{i}'
            object_num = getattr(data_obj, 'num', None)
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = getattr(data_obj, 'keepout')
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(data_obj, 'locations', [])
            object_placements = getattr(data_obj, 'placements', None)
            object_keepout = getattr(data_obj, 'keepout')
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]  # pylint: disable=invalid-name
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensor's data correct
        obs = {}

        obs.update(self.agent.obs_sensor())

        for obstacle in self._obstacles:
            if obstacle.is_lidar_observed:
                obs[obstacle.name + '_lidar'] = self._obs_lidar(obstacle.pos, obstacle.group)
            if hasattr(obstacle, 'is_comp_observed') and obstacle.is_comp_observed:
                obs[obstacle.name + '_comp'] = self._obs_compass(obstacle.pos)

        if self.observe_vision:
            obs['vision'] = self._obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_info.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_info.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset : offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
            assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
            assert (
                offset == self.obs_info.obs_flat_size
            ), 'Obs from mujoco do not match env pre-specifed lenth.'
        return obs

    def _obs_lidar(self, positions, group):
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_conf.type == 'pseudo':
            return self._obs_lidar_pseudo(positions)

        if self.lidar_conf.type == 'natural':
            return self._obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_conf.type}')

    def _obs_lidar_natural(self, group):
        """Natural lidar casts rays based on the ego-frame of the agent.

        Rays are circularly projected from the agent body origin around the agent z axis.
        """
        body = self.model.body('agent').id
        # pylint: disable-next=no-member
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.agent.pos, dtype='float64')
        mat_t = self.agent.mat
        obs = np.zeros(self.lidar_conf.num_bins)
        for i in range(self.lidar_conf.num_bins):
            theta = (i / self.lidar_conf.num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            geom_id = np.array([0], dtype='int32')
            dist = mujoco.mj_ray(  # pylint: disable=no-member
                self.model, self.data, pos, vec, grp, 1, body, geom_id
            )
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def _obs_lidar_pseudo(self, positions):
        """Return a agent-centric lidar observation of a list of positions.

        Lidar is a set of bins around the agent (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the agent.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the agent)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        """
        positions = np.array(positions, ndmin=2)
        obs = np.zeros(self.lidar_conf.num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            # pylint: disable-next=invalid-name
            z = complex(*self._ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_conf.num_bins
            bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
            bin_angle = bin_size * bin
            if self.lidar_conf.max_dist is None:
                sensor = np.exp(-self.lidar_conf.exp_gain * dist)
            else:
                sensor = max(0, self.lidar_conf.max_dist - dist) / self.lidar_conf.max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_conf.alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_conf.num_bins
                bin_minus = (bin - 1) % self.lidar_conf.num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def _obs_compass(self, pos):
        """Return a agent-centric compass observation of a list of positions.

        Compass is a normalized (unit-length) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        # Get ego vector in world frame
        vec = pos - self.agent.pos
        # Rotate into frame
        vec = np.matmul(vec, self.agent.mat)
        # Truncate
        vec = vec[: self.compass_conf.shape]
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        assert vec.shape == (self.compass_conf.shape,), f'Bad vec {vec}'
        return vec

    def _obs_vision(self):
        """Return pixels from the agent camera."""
        rows, cols = self.vision_env_conf.vision_size
        width, height = cols, rows
        vision = self.render(width, height, mode='rgb_array', camera_name='vision', cost={})
        return vision

    def _ego_xy(self, pos):
        """Return the egocentric XY vector to a position from the agent."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        agent_3vec = self.agent.pos
        agent_mat = self.agent.mat
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - agent_3vec
        return np.matmul(world_3vec, agent_mat)[:2]  # only take XY coordinates

    @abc.abstractmethod
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""

    @abc.abstractmethod
    def specific_reset(self):
        """Set positions and orientations of agent and obstacles."""

    @abc.abstractmethod
    def specific_step(self):
        """Each task can define a specific step function.

        It will be called when :func:`step()` is called using env.step().
        For example, you can do specific data modification.
        """

    @abc.abstractmethod
    def update_world(self):
        """Update one task specific goal."""

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""

    @property
    @abc.abstractmethod
    def goal_achieved(self):
        """Check if task specific goal is achieved."""
