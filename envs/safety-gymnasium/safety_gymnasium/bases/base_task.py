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
from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from safety_gymnasium.assets.geoms import GEOMS_REGISTER
from safety_gymnasium.assets.mocaps import MOCAPS_REGISTER
from safety_gymnasium.assets.objects import OBJS_REGISTER
from safety_gymnasium.assets.robot import Robot
from safety_gymnasium.bases import BaseMujocoTask
from safety_gymnasium.utils.common_utils import ResamplingError
from safety_gymnasium.utils.task_utils import quat2mat, theta2vec


class BaseTask(BaseMujocoTask):
    """Base task class for defining some common characteristic."""

    def __init__(self, config: dict):  # pylint: disable-next=too-many-statements
        super().__init__(config=config)

        self.num_steps = 1000  # Maximum number of environment steps in an episode

        self.placements_extents = [-2, -2, 2, 2]  # Placement limits (min X, min Y, max X, max Y)
        self.placements_margin = 0.0  # Additional margin added to keepout when placing objects
        # Starting position distribution
        self.randomize_layout = True  # If false, set the random seed before layout to constant
        self.build_resample = True  # If true, rejection sample from valid environments
        self.continue_goal = True  # If true, draw a new goal after achievement
        self.terminate_resample_failure = True  # If true, end episode when resampling fails,
        # otherwise, raise a python exception.

        # Sensor observations
        # Specify which sensors to add to observation space
        self.sensors_obs = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
        self.sensors_hinge_joints = True  # Observe named joint position / velocity sensors
        self.sensors_ball_joints = True  # Observe named balljoint position / velocity sensors
        self.sensors_angle_components = True  # Observe sin/cos theta instead of theta

        # Lidar observation parameters
        self.lidar_num_bins = 16  # Bins (around a full circle) for lidar sensing
        self.lidar_max_dist = (
            3  # Maximum distance for lidar sensitivity (if None, exponential distance)
        )
        self.lidar_exp_gain = 1.0  # Scaling factor for distance in exponential distance lidar
        self.lidar_type = 'pseudo'  # 'pseudo', 'natural', see self.obs_lidar()
        self.lidar_alias = True  # Lidar bins alias into each other

        # Compass observation parameters
        self.compass_shape = 2  # Set to 2 or 3 for XY or XYZ unit vector compass observation.

        self.reward_orientation = False  # Reward for being upright
        self.reward_orientation_scale = 0.002  # Scale for uprightness reward
        self.reward_orientation_body = 'robot'  # What body to get orientation from
        self.reward_exception = -10.0  # Reward when encoutering a mujoco exception
        self.reward_z = 1.0  # Reward for standup tests (vel in z direction)
        self.reward_clip = (
            10  # Clip reward, last resort against physics errors causing magnitude spikes
        )

        self.constrain_indicator = True  # If true, all costs are either 1 or 0 for a given step.

        self._seed = None  # Random state seed (avoid name conflict with self.seed)

        self.robot = Robot(self.robot_base)  # pylint: disable=no-member
        self.action_space = spaces.Box(-1, 1, (self.robot.nu,), dtype=np.float64)
        self.action_noise = 0.0  # Magnitude of independent per-component gaussian action noise
        self.observe_vision = False  # Observe vision from the robot
        self.observation_flatten = True  # Flatten observation into a vector
        self.obs_flat_size = None

        # Obstacles which are added in environments.
        self._geoms = {}
        self._objects = {}
        self._mocaps = {}

    def add_geoms(self, *geoms):
        """Add geom type objects into environments and set coresponding attributes."""
        for geom in geoms:
            assert (
                type(geom) in GEOMS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._geoms[geom.name] = geom
            setattr(self, geom.name, geom)

    def add_objects(self, *objects):
        """Add object type objects into environments and set coresponding attributes."""
        for obj in objects:
            assert (
                type(obj) in OBJS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._objects[obj.name] = obj
            setattr(self, obj.name, obj)

    def add_mocaps(self, *mocaps):
        """Add mocap type objects into environments and set coresponding attributes."""
        for mocap in mocaps:
            assert (
                type(mocap) in MOCAPS_REGISTER
            ), 'Please figure out the type of object before you add it into envs.'
            self._mocaps[mocap.name] = mocap
            setattr(self, mocap.name, mocap)

    def dist_goal(self):
        """Return the distance from the robot to the goal XY position."""
        return self.dist_xy(self.goal_pos[0])

    def dist_xy(self, pos):
        """Return the distance from the robot to an XY position."""
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))

    def calculate_cost(self):
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}
        # Calculate constraint violations
        for geom in self._geoms.values():
            if geom.is_constrained:
                cost.update(geom.cal_cost(engine=self))
        for obj in self._objects.values():
            if obj.is_constrained:
                cost.update(obj.cal_cost(engine=self))
        for mocap in self._mocaps.values():
            if mocap.is_constrained:
                cost.update(mocap.cal_cost(engine=self))

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        return cost

    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__ in Builder."""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.build_sensor_observation_space())

        for geom in self._geoms.values():
            name = geom.name + '_' + 'lidar'
            obs_space_dict[name] = gymnasium.spaces.Box(
                0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
            )
        for obj in self._objects.values():
            name = obj.name + '_' + 'lidar'
            obs_space_dict[name] = gymnasium.spaces.Box(
                0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
            )
        for mocap in self._mocaps.values():
            name = mocap.name + '_' + 'lidar'
            obs_space_dict[name] = gymnasium.spaces.Box(
                0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
            )

        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gymnasium.spaces.Box(
                0, 255, self.vision_size + (3,), dtype=np.uint8
            )

        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum(np.prod(i.shape) for i in self.obs_space_dict.values())
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def build_sensor_observation_space(self):
        """Build observation space for all sensor types."""
        obs_space_dict = {}

        for sensor in self.sensors_obs:  # Explicitly listed sensors
            dim = self.robot.sensor_dim[sensor]
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (dim,), dtype=np.float64)
        # Velocities don't have wraparound effects that rotational positions do
        # Wraparounds are not kind to neural networks
        # Whereas the angle 2*pi is very close to 0, this isn't true in the network
        # In theory the network could learn this, but in practice we simplify it
        # when the sensors_angle_components switch is enabled.
        for sensor in self.robot.hinge_vel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float64)
        for sensor in self.robot.ballangvel_names:
            obs_space_dict[sensor] = gymnasium.spaces.Box(-np.inf, np.inf, (3,), dtype=np.float64)
        # Angular positions have wraparound effects, so output something more friendly
        if self.sensors_angle_components:
            # Single joints are turned into sin(x), cos(x) pairs
            # These should be easier to learn for neural networks,
            # Since for angles, small perturbations in angle give small differences in sin/cos
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (2,), dtype=np.float64
                )
            # Quaternions are turned into 3x3 rotation matrices
            # Quaternions have a wraparound issue in how they are normalized,
            # where the convention is to change the sign so the first element to be positive.
            # If the first element is close to 0, this can mean small differences in rotation
            # lead to large differences in value as the latter elements change sign.
            # This also means that the first element of the quaternion is not expectation zero.
            # The SO(3) rotation representation would be a good replacement here,
            # since it smoothly varies between values in all directions (the property we want),
            # but right now we have very little code to support SO(3) roatations.
            # Instead we use a 3x3 rotation matrix, which if normalized, smoothly varies as well.
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (3, 3), dtype=np.float64
                )
        else:
            # Otherwise include the sensor without any processing
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float64
                )
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (4,), dtype=np.float64
                )

        return obs_space_dict

    def build_placements_dict(self):
        """Build a dict of placements.

        Happens only once.
        """
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))

        for geom in self._geoms.values():
            placements.update(self.placements_dict_from_object(geom.name))
        for obj in self._objects.values():
            placements.update(self.placements_dict_from_object(obj.name))
        for mocap in self._mocaps.values():
            placements.update(self.placements_dict_from_object(mocap.name))
        self.placements = placements

    def toggle_observation_space(self):
        """Toggle observation space."""
        self.observation_flatten = not self.observation_flatten
        self.build_observation_space()

    def build_world_config(self, layout):
        """Create a world_config from our own config"""
        world_config = {}

        world_config['robot_base'] = self.robot.base
        world_config['robot_xy'] = layout['robot']
        if self.robot.rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot.rot)

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        for geom in self._geoms.values():
            if hasattr(geom, 'num'):
                for i in range(geom.num):
                    name = f'{geom.name[:-1]}{i}'
                    world_config['geoms'][name] = geom.get(
                        index=i, layout=layout, rot=self.random_rot()
                    )
            else:
                world_config['geoms'][geom.name] = geom.get(layout=layout, rot=self.random_rot())

        # Extra objects to add to the scene
        world_config['objects'] = {}
        for obj in self._objects.values():
            if hasattr(obj, 'num'):
                for i in range(obj.num):
                    name = f'{obj.name[:-1]}{i}'
                    world_config['objects'][name] = obj.get(
                        index=i, layout=layout, rot=self.random_rot()
                    )
            else:
                world_config['objects'][obj.name] = obj.get(layout=layout, rot=self.random_rot())

        # Extra mocap bodies used for control (equality to object of same name)
        world_config['mocaps'] = {}
        for mocap in self._mocaps.values():
            if hasattr(mocap, 'num'):
                for i in range(mocap.num):
                    mocap_name = f'{mocap.name[:-1]}{i}mocap'
                    obj_name = f'{mocap.name[:-1]}{i}obj'
                    rot = self.random_rot()
                    world_config['objects'][obj_name] = mocap.get_obj(
                        index=i, layout=layout, rot=rot
                    )
                    world_config['mocaps'][mocap_name] = mocap.get_mocap(
                        index=i, layout=layout, rot=rot
                    )
            else:
                mocap_name = f'{mocap.name[:-1]}mocap'
                obj_name = f'{mocap.name[:-1]}obj'
                rot = self.random_rot()
                world_config['objects'][obj_name] = mocap.get_obj(index=i, layout=layout, rot=rot)
                world_config['mocaps'][mocap_name] = mocap.get_mocap(
                    index=i, layout=layout, rot=rot
                )

        return world_config

    def build_goal_position(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # Resample until goal is compatible with layout
        if 'goal' in self.layout:
            del self.layout['goal']
        for _ in range(10000):  # Retries
            if self.sample_goal_position():
                break
        else:
            raise ResamplingError('Failed to generate goal')
        # Move goal geom to new layout position
        self.world_config_dict['geoms']['goal']['pos'][:2] = self.layout['goal']

        goal_body_id = self.model.body('goal').id
        self.model.body(goal_body_id).pos[:2] = self.layout['goal']
        mujoco.mj_forward(self.model, self.data)  # pylint: disable=no-member

    def placements_dict_from_object(self, object_name):
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
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensordata correct
        obs = {}

        obs.update(self.obs_sensor())

        for geom in self._geoms.values():
            name = geom.name + '_' + 'lidar'
            obs[name] = self.obs_lidar(getattr(self, geom.name + '_pos'), geom.group)
        for obj in self._objects.values():
            name = obj.name + '_' + 'lidar'
            obs[name] = self.obs_lidar(getattr(self, obj.name + '_pos'), obj.group)
        for mocap in self._mocaps.values():
            name = mocap.name + '_' + 'lidar'
            obs[name] = self.obs_lidar(getattr(self, mocap.name + '_pos'), mocap.group)

        if self.observe_vision:
            obs['vision'] = self.obs_vision()
        if self.observation_flatten:
            flat_obs = np.zeros(self.obs_flat_size)
            offset = 0
            for k in sorted(self.obs_space_dict.keys()):
                k_size = np.prod(obs[k].shape)
                flat_obs[offset : offset + k_size] = obs[k].flat
                offset += k_size
            obs = flat_obs
            assert self.observation_space.contains(obs), f'Bad obs {obs} {self.observation_space}'
            assert (
                offset == self.obs_flat_size
            ), 'Obs from mujoco do not match env pre-specifed lenth.'
        return obs

    def obs_sensor(self):
        """Get observations of all sensor types."""
        obs = {}

        # Sensors which can be read directly, without processing
        for sensor in self.sensors_obs:  # Explicitly listed sensors
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.hinge_vel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        for sensor in self.robot.ballangvel_names:
            obs[sensor] = self.world.get_sensor(sensor)
        # Process angular position sensors
        if self.sensors_angle_components:
            for sensor in self.robot.hinge_pos_names:
                theta = float(self.world.get_sensor(sensor))  # Ensure not 1D, 1-element array
                obs[sensor] = np.array([np.sin(theta), np.cos(theta)])
            for sensor in self.robot.ballquat_names:
                quat = self.world.get_sensor(sensor)
                obs[sensor] = quat2mat(quat)
        else:  # Otherwise read sensors directly
            for sensor in self.robot.hinge_pos_names:
                obs[sensor] = self.world.get_sensor(sensor)
            for sensor in self.robot.ballquat_names:
                obs[sensor] = self.world.get_sensor(sensor)

        return obs

    def obs_lidar(self, positions, group):
        """Calculate and return a lidar observation.

        See sub methods for implementation.
        """
        if self.lidar_type == 'pseudo':
            return self.obs_lidar_pseudo(positions)

        if self.lidar_type == 'natural':
            return self.obs_lidar_natural(group)

        raise ValueError(f'Invalid lidar_type {self.lidar_type}')

    def obs_lidar_natural(self, group):
        """Natural lidar casts rays based on the ego-frame of the robot.

        Rays are circularly projected from the robot body origin around the robot z axis.
        """
        body = self.model.body_name2id('robot')
        # pylint: disable-next=no-member
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.world.robot_pos(), dtype='float64')
        mat_t = self.world.robot_mat()
        obs = np.zeros(self.lidar_num_bins)
        for i in range(self.lidar_num_bins):
            theta = (i / self.lidar_num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)  # pylint: disable=no-member
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def obs_lidar_pseudo(self, positions):
        """Return a robot-centric lidar observation of a list of positions.

        Lidar is a set of bins around the robot (divided evenly in a circle).
        The detection directions are exclusive and exhaustive for a full 360 view.
        Each bin reads 0 if there are no objects in that direction.
        If there are multiple objects, the distance to the closest one is used.
        Otherwise the bin reads the fraction of the distance towards the robot.

        E.g. if the object is 90% of lidar_max_dist away, the bin will read 0.1,
        and if the object is 10% of lidar_max_dist away, the bin will read 0.9.
        (The reading can be thought of as "closeness" or inverse distance)

        This encoding has some desirable properties:
            - bins read 0 when empty
            - bins smoothly increase as objects get close
            - maximum reading is 1.0 (where the object overlaps the robot)
            - close objects occlude far objects
            - constant size observation with variable numbers of objects
        """
        obs = np.zeros(self.lidar_num_bins)
        for pos in positions:
            pos = np.asarray(pos)
            if pos.shape == (3,):
                pos = pos[:2]  # Truncate Z coordinate
            # pylint: disable-next=invalid-name
            z = complex(*self.ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_num_bins
            bin = int(angle / bin_size)  # pylint: disable=redefined-builtin
            bin_angle = bin_size * bin
            if self.lidar_max_dist is None:
                sensor = np.exp(-self.lidar_exp_gain * dist)
            else:
                sensor = max(0, self.lidar_max_dist - dist) / self.lidar_max_dist
            obs[bin] = max(obs[bin], sensor)
            # Aliasing
            if self.lidar_alias:
                alias = (angle - bin_angle) / bin_size
                assert 0 <= alias <= 1, f'bad alias {alias}, dist {dist}, angle {angle}, bin {bin}'
                bin_plus = (bin + 1) % self.lidar_num_bins
                bin_minus = (bin - 1) % self.lidar_num_bins
                obs[bin_plus] = max(obs[bin_plus], alias * sensor)
                obs[bin_minus] = max(obs[bin_minus], (1 - alias) * sensor)
        return obs

    def obs_compass(self, pos):
        """Return a robot-centric compass observation of a list of positions.

        Compass is a normalized (unit-lenght) egocentric XY vector,
        from the agent to the object.

        This is equivalent to observing the egocentric XY angle to the target,
        projected into the sin/cos space we use for joints.
        (See comment on joint observation for why we do this.)
        """
        pos = np.asarray(pos)
        if pos.shape == (2,):
            pos = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        # Get ego vector in world frame
        vec = pos - self.world.robot_pos()
        # Rotate into frame
        vec = np.matmul(vec, self.world.robot_mat())
        # Truncate
        vec = vec[: self.compass_shape]
        # Normalize
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        assert vec.shape == (self.compass_shape,), f'Bad vec {vec}'
        return vec

    def obs_vision(self):
        """Return pixels from the robot camera."""
        rows, cols = self.vision_size
        width, height = cols, rows
        vision = self.render(width, height, mode='rgb_array', camera_name='vision', cost={})
        return vision

    def ego_xy(self, pos):
        """Return the egocentric XY vector to a position from the robot."""
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates

    def random_rot(self):
        """Use internal random state to get a random rotation in radians."""
        return self.random_generator.uniform(0, 2 * np.pi)

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""

    @property
    def robot_pos(self):
        """Helper to get current robot position."""
        return self.data.body('robot').xpos.copy()

    @property
    def walls_pos(self):
        """Helper to get the walls positions from layout."""
        # pylint: disable-next=no-member
        return [self.data.body(f'wall{i}').xpos.copy() for i in range(self.walls_num)]

    @property
    @abc.abstractmethod
    def goal_achieved(self):
        """Check if task specific goal is achieved."""

    @abc.abstractmethod
    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""

    @abc.abstractmethod
    def specific_agent_config(self):
        """Modify the agents property according to task.

        In some tasks, agents should be modified a little.
        """

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
    def build_goal(self):
        """Update one task specific goal."""

    @abc.abstractmethod
    def update_world(self):
        """Some Tasks will update the world after achieving part of goals."""
