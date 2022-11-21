import abc
from collections import OrderedDict
from copy import deepcopy

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.robot import Robot
from safety_gymnasium.envs.safety_gym_v2.utils import quat2mat, theta2vec


class BaseTask(abc.ABC):
    def __init__(
        self,
        task_config: dict,
    ):

        self.num_steps = 1000  # Maximum number of environment steps in an episode

        self.action_noise = 0.0  # Magnitude of independent per-component gaussian action noise

        self.placements_extents = [
            -2,
            -2,
            2,
            2,
        ]  # Placement limits (min X, min Y, max X, max Y)
        self.placements_margin = 0.0  # Additional margin added to keepout when placing objects

        self.floor_display_mode = False  # In display mode, the visible part of the floor is cropped
        # self.floor_size = [3.5, 3.5, .1]

        self.robot_placements = None  # Robot placements list (defaults to full extents)
        self.robot_locations = []  # Explicitly place robot XY coordinate
        self.robot_keepout = 0.4  # Needs to be set to match the robot XML used
        self.robot_base = 'xmls/car.xml'  # Which robot XML to use as the base
        self.robot_rot = None  # Override robot starting angle

        # Starting position distribution
        self.randomize_layout = True  # If false, set the random seed before layout to constant
        self.build_resample = True  # If true, rejection sample from valid environments
        self.continue_goal = True  # If true, draw a new goal after achievement
        self.terminate_resample_failure = True  # If true, end episode when resampling fails,
        # otherwise, raise a python exception.
        # TODO: randomize starting joint positions

        self.observe_vision = False
        self.observation_flatten = True

        # # Vision observation parameters
        # TODO JJM
        self.vision_size = (
            60,
            40,
        )  # Size (width, height) of vision observation; gets flipped internally to (rows, cols) format
        self.vision_render = True  # Render vision observation in the viewer
        self.vision_render_size = (300, 200)  # Size to render the vision in the viewer

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

        # Goal parameters
        self.goal_placements = None  # Placements where goal may appear (defaults to full extents)
        self.goal_locations = []  # Fixed locations to override placements
        self.goal_keepout = 0.4  # Keepout radius when placing goals

        self.apples_num = 0
        self.apples_placements = None
        self.apples_locations = []
        self.apples_keepout = 0.3

        self.oranges_num = 0
        self.oranges_placements = None
        self.oranges_locations = []
        self.oranges_keepout = 0.3

        # Box parameters (only used if task == 'push')
        self.box_placements = None  # Box placements list (defaults to full extents)
        self.box_locations = []  # Fixed locations to override placements
        self.box_keepout = 0.2  # Box keepout radius for placement
        self.box_null_dist = 2  # Within box_null_dist * box_size radius of box, no box reward given

        self.reward_goal = 1.0  # Sparse reward for being inside the goal area
        self.reward_orientation = False  # Reward for being upright
        self.reward_orientation_scale = 0.002  # Scale for uprightness reward
        self.reward_orientation_body = 'robot'  # What body to get orientation from
        self.reward_exception = -10.0  # Reward when encoutering a mujoco exception
        self.reward_z = 1.0  # Reward for standup tests (vel in z direction)
        self.reward_clip = (
            10  # Clip reward, last resort against physics errors causing magnitude spikes
        )

        # Buttons are small immovable spheres, to the environment
        self.buttons_num = 0  # Number of buttons to add
        self.buttons_placements = None  # Buttons placements list (defaults to full extents)
        self.buttons_locations = []  # Fixed locations to override placements
        self.buttons_keepout = 0.3  # Buttons keepout radius for placement
        self.buttons_cost = 1.0  # Cost for pressing the wrong button, if constrain_buttons
        self.buttons_resampling_delay = (
            10  # Buttons have a timeout period (steps) before resampling
        )

        # Sensor observations
        # Specify which sensors to add to observation space
        self.sensors_obs = (['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],)
        self.sensors_hinge_joints = True  # Observe named joint position / velocity sensors
        self.sensors_ball_joints = True  # Observe named balljoint position / velocity sensors
        self.sensors_angle_components = True  # Observe sin/cos theta instead of theta

        # Walls - barriers in the environment not associated with any constraint
        # NOTE: this is probably best to be auto-generated than manually specified
        self.walls_num = 0  # Number of walls
        self.walls_placements = None  # This should not be used
        self.walls_locations = []  # This should be used and length == walls_num
        self.walls_keepout = 0.0  # This should not be used

        self.constrain_indicator = True  # If true, all costs are either 1 or 0 for a given step.

        # Hazardous areas
        self.hazards_num = 0  # Number of hazards in an environment
        self.hazards_placements = None  # Placements list for hazards (defaults to full extents)
        self.hazards_locations = []  # Fixed locations to override placements
        self.hazards_keepout = 0.4  # Radius of hazard keepout for placement
        self.hazards_cost = 1.0  # Cost (per step) for violating the constraint

        # Vases (objects we should not touch)
        self.vases_num = 0  # Number of vases in the world
        self.vases_placements = None  # Vases placements list (defaults to full extents)
        self.vases_locations = []  # Fixed locations to override placements
        self.vases_keepout = 0.15  # Radius of vases keepout for placement
        self.vases_sink = 4e-5  # Experimentally measured, based on size and density,
        # how far vases "sink" into the floor.
        # Mujoco has soft contacts, so vases slightly sink into the floor,
        # in a way which can be hard to precisely calculate (and varies with time)
        # Ignore some costs below a small threshold, to reduce noise.
        self.vases_contact_cost = 1.0  # Cost (per step) for being in contact with a vase
        self.vases_displace_cost = 0.0  # Cost (per step) per meter of displacement for a vase
        self.vases_displace_threshold = 1e-3  # Threshold for displacement being "real"
        self.vases_velocity_cost = 1.0  # Cost (per step) per m/s of velocity for a vase
        self.vases_velocity_threshold = 1e-4  # Ignore very small velocities

        # Pillars (immovable obstacles we should not touch)
        self.pillars_num = 0  # Number of pillars in the world
        self.pillars_placements = None  # Pillars placements list (defaults to full extents)
        self.pillars_locations = []  # Fixed locations to override placements
        self.pillars_keepout = 0.3  # Radius for placement of pillars
        self.pillars_cost = 1.0  # Cost (per step) for being in contact with a pillar

        # Gremlins (moving objects we should avoid)
        self.gremlins_num = 0  # Number of gremlins in the world
        self.gremlins_placements = None  # Gremlins placements list (defaults to full extents)
        self.gremlins_locations = []  # Fixed locations to override placements
        self.gremlins_keepout = 0.5  # Radius for keeping out (contains gremlin path)
        self.gremlins_travel = 0.3  # Radius of the circle traveled in
        self.gremlins_contact_cost = 1.0  # Cost for touching a gremlin
        self.gremlins_dist_threshold = 0.2  # Threshold for cost for being too close
        self.gremlins_dist_cost = 1.0  # Cost for being within distance threshold

        self._seed = None  # Random state seed (avoid name conflict with self.seed)

        self.parse(task_config)
        self.robot = Robot(self.robot_base)
        self.build_observation_space()

    def parse(self, config):
        """Parse a config dict - see self.DEFAULT for description"""
        self.config = {}
        self.config.update(deepcopy(config))
        for key, value in self.config.items():
            setattr(self, key, value)

    def set_engine(self, engine):
        self.engine = engine

    def specific_step(self):
        pass

    def task_continue_reset(self):
        pass

    @abc.abstractmethod
    def calculate_cost(self):
        """determine costs depending on agent and obstacles"""
        raise NotImplementedError

    @abc.abstractmethod
    def calculate_reward(self):
        """Implements the task's specific reward function, which depends on
        the agent and the surrounding obstacles."""
        raise NotImplementedError

    def build_goal(self):
        """Update one task specific goal."""
        pass

    def update_world(self):
        """Some Tasks will update the world after achieving some goals."""
        pass

    @abc.abstractmethod
    def agent_specific_config(self):
        raise NotImplementedError

    @abc.abstractmethod
    def specific_reset(self):
        """Set positions and orientations of agent and obstacles."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def goal_achieved(self):
        """Check if task specific goal is achieved."""
        raise NotImplementedError

    @property
    def goal_pos(self):
        """Helper to get goal position from layout"""
        pass

    @property
    def world(self):
        return self.engine.world

    @property
    def model(self):
        """Helper to get the world's model instance"""
        return self.world.model

    @property
    def data(self):
        """Helper to get the world's simulation data instance"""
        return self.world.data

    @property
    def robot_pos(self):
        """Helper to get current robot position"""
        return self.data.body('robot').xpos.copy()

    @property
    def walls_pos(self):
        """Helper to get the hazards positions from layout"""
        return [self.data.body(f'wall{i}').xpos.copy() for i in range(self.walls_num)]

    @property
    def rs(self):
        return self.engine.rs

    def dist_goal(self):
        """Return the distance from the robot to the goal XY position"""
        return self.dist_xy(self.goal_pos)

    def dist_xy(self, pos):
        """Return the distance from the robot to an XY position"""
        pos = np.asarray(pos)
        if pos.shape == (3,):
            pos = pos[:2]
        robot_pos = self.world.robot_pos()
        return np.sqrt(np.sum(np.square(pos - robot_pos[:2])))

    @abc.abstractmethod
    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__"""
        pass

    @abc.abstractmethod
    def build_placements_dict(self):
        """Build a dict of placements.  Happens once during __init__."""
        raise NotImplementedError

    def placements_dict_from_object(self, object_name):
        """Get the placements dict subset just for a given object name"""
        placements_dict = {}
        if hasattr(self, object_name + 's_num'):  # Objects with multiplicity
            plural_name = object_name + 's'
            object_fmt = object_name + '{i}'
            object_num = getattr(self, plural_name + '_num', None)
            object_locations = getattr(self, plural_name + '_locations', [])
            object_placements = getattr(self, plural_name + '_placements', None)
            object_keepout = getattr(self, plural_name + '_keepout')
        else:  # Unique objects
            object_fmt = object_name
            object_num = 1
            object_locations = getattr(self, object_name + '_locations', [])
            object_placements = getattr(self, object_name + '_placements', None)
            object_keepout = getattr(self, object_name + '_keepout')
        for i in range(object_num):
            if i < len(object_locations):
                x, y = object_locations[i]
                k = object_keepout + 1e-9  # Epsilon to account for numerical issues
                placements = [(x - k, y - k, x + k, y + k)]
            else:
                placements = object_placements
            placements_dict[object_fmt.format(i=i)] = (placements, object_keepout)
        return placements_dict

    @abc.abstractmethod
    def build_world_config(self, layout):
        """Create a world_config from our own config"""
        pass

    @abc.abstractmethod
    def obs(self):
        """Return the observation of our agent"""
        pass

    def set_mocaps(self):
        """Set mocap object positions before a physics step is executed"""
        # if self.gremlins_num: # self.constrain_gremlins:
        #     phase = float(self.data.time)
        #     for i in range(self.gremlins_num):
        #         name = f'gremlin{i}'
        #         target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins_travel
        #         pos = np.r_[target, [self.gremlins_size]]
        #         self.data.set_mocap_pos(name + 'mocap', pos)
        pass

    def random_rot(self):
        """Use internal random state to get a random rotation in radians"""
        return self.rs.uniform(0, 2 * np.pi)

    def obs_compass(self, pos):
        """
        Return a robot-centric compass observation of a list of positions.

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
        """Return pixels from the robot camera"""
        # Get a render context so we can
        rows, cols = self.vision_size
        width, height = cols, rows
        vision = self.engine.render(width, height, mode='rgb_array', camera_name='vision', cost={})
        return vision

    def obs_lidar(self, positions, group):
        """
        Calculate and return a lidar observation.  See sub methods for implementation.
        """
        if self.lidar_type == 'pseudo':
            return self.obs_lidar_pseudo(positions)
        elif self.lidar_type == 'natural':
            return self.obs_lidar_natural(group)
        else:
            raise ValueError(f'Invalid lidar_type {self.lidar_type}')

    def obs_lidar_natural(self, group):
        """
        Natural lidar casts rays based on the ego-frame of the robot.
        Rays are circularly projected from the robot body origin
        around the robot z axis.
        """
        body = self.model.body_name2id('robot')
        grp = np.asarray([i == group for i in range(int(mujoco.mjNGROUP))], dtype='uint8')
        pos = np.asarray(self.world.robot_pos(), dtype='float64')
        mat_t = self.world.robot_mat()
        obs = np.zeros(self.lidar_num_bins)
        for i in range(self.lidar_num_bins):
            theta = (i / self.lidar_num_bins) * np.pi * 2
            vec = np.matmul(mat_t, theta2vec(theta))  # Rotate from ego to world frame
            vec = np.asarray(vec, dtype='float64')
            dist, _ = self.sim.ray_fast_group(pos, vec, grp, 1, body)
            if dist >= 0:
                obs[i] = np.exp(-dist)
        return obs

    def obs_lidar_pseudo(self, positions):
        """
        Return a robot-centric lidar observation of a list of positions.

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
            z = complex(*self.ego_xy(pos))  # X, Y as real, imaginary components
            dist = np.abs(z)
            angle = np.angle(z) % (np.pi * 2)
            bin_size = (np.pi * 2) / self.lidar_num_bins
            bin = int(angle / bin_size)
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

    def ego_xy(self, pos):
        """Return the egocentric XY vector to a position from the robot"""
        assert pos.shape == (2,), f'Bad pos {pos}'
        robot_3vec = self.world.robot_pos()
        robot_mat = self.world.robot_mat()
        pos_3vec = np.concatenate([pos, [0]])  # Add a zero z-coordinate
        world_3vec = pos_3vec - robot_3vec
        return np.matmul(world_3vec, robot_mat)[:2]  # only take XY coordinates

    def get_sensor_obs(self):
        obs = {}

        # if self.observe_sensors:
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

    def build_sensor_observation_space(self):
        obs_space_dict = {}

        # if self.observe_sensors:
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
            # TODO: comparative study of the performance with and without this feature.
            for sensor in self.robot.hinge_pos_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (1,), dtype=np.float64
                )
            for sensor in self.robot.ballquat_names:
                obs_space_dict[sensor] = gymnasium.spaces.Box(
                    -np.inf, np.inf, (4,), dtype=np.float64
                )

        return obs_space_dict
