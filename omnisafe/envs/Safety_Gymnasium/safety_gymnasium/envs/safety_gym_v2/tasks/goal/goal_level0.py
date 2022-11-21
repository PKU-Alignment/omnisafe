"""goal_level0.py ends here"""
from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.goal import get_goal
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP
from safety_gymnasium.envs.safety_gym_v2.base_task import BaseTask
from safety_gymnasium.envs.safety_gym_v2.utils import quat2mat


class GoalLevel0(BaseTask):
    """A task where agents have to run as fast as possible within a circular
    zone.
    Rewards are by default shaped.

    """

    def __init__(
        self,
        task_config,
    ):
        super().__init__(
            task_config=task_config,
        )

        self.placements_extents = [-1, -1, 1, 1]

        self.goal_size = 0.3
        self.goal_keepout = 0.305

        # Reward is distance towards goal plus a constant for being within range of goal
        # reward_distance should be positive to encourage moving towards the goal
        # if reward_distance is 0, then the reward function is sparse
        self.reward_distance = 1.0  # Dense reward multiplied by the distance moved to the goal

        self.hazards_size = 0.2
        self.hazards_keepout = 0.18

        self.agent_specific_config()
        self.last_dist_goal = None

    # pylint: disable=W0613
    def calculate_cost(self, **kwargs):
        """determine costs depending on agent and obstacles"""
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost

    def calculate_reward(self):
        """Returns the reward of an agent running in a circle (clock-wise)."""
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.reward_goal

        return reward

    @property
    def goal_achieved(self):
        # agent runs endlessly
        return self.dist_goal() <= self.goal_size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout"""
        return self.data.body('goal').xpos.copy()

    @property
    def hazards_pos(self):
        """Helper to get the hazards positions from layout"""
        return [self.data.body(f'hazard{i}').xpos.copy() for i in range(self.hazards_num)]

    @property
    def vases_pos(self):
        """Helper to get the list of vase positions"""
        return [self.data.body(f'vase{p}').xpos.copy() for p in range(self.vases_num)]

    def agent_specific_config(self):
        pass

    # pylint: disable=W0107
    def specific_reset(self):
        """Reset agent position and set orientation towards desired run
        direction."""
        pass

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards"""
        self.engine.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def build_placements_dict(self):
        """Build a dict of placements.  Happens once during __init__."""
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))

        placements.update(self.placements_dict_from_object('goal'))

        return placements

    def build_world_config(self, layout):
        """Create a world_config from our own config"""
        # TODO: parse into only the pieces we want/need
        world_config = {}

        world_config['robot_base'] = self.robot_base
        world_config['robot_xy'] = layout['robot']
        if self.robot_rot is None:
            world_config['robot_rot'] = self.random_rot()
        else:
            world_config['robot_rot'] = float(self.robot_rot)

        # if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        # world_config['observe_vision'] = self.observe_vision

        # Extra objects to add to the scene
        world_config['objects'] = {}

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        world_config['geoms']['goal'] = get_goal(
            layout=layout, rot=self.random_rot(), size=self.goal_size
        )

        return world_config

    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__"""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.build_sensor_observation_space())

        # if self.observe_goal_lidar:
        obs_space_dict['goal_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
        )

        if self.observe_vision:
            width, height = self.vision_size
            rows, cols = height, width
            self.vision_size = (rows, cols)
            obs_space_dict['vision'] = gymnasium.spaces.Box(0, 255, self.vision_size + (3,), dtype=np.uint8)

        # Flatten it ourselves
        self.obs_space_dict = obs_space_dict
        if self.observation_flatten:
            self.obs_flat_size = sum([np.prod(i.shape) for i in self.obs_space_dict.values()])
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def obs(self):
        """Return the observation of our agent"""
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensordata correct
        obs = {}

        # if self.observe_goal_lidar:
        obs['goal_lidar'] = self.obs_lidar([self.goal_pos], GROUP['goal'])

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
        return obs
