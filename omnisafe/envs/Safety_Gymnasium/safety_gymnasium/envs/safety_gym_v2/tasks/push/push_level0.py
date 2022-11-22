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

"""push_level0"""
from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.goal import get_goal
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP
from safety_gymnasium.envs.safety_gym_v2.assets.push_box import get_push_box
from safety_gymnasium.envs.safety_gym_v2.base_task import BaseTask


class PushLevel0(BaseTask):
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
        self.goal_size = 0.3

        self.placements_extents = [-1, -1, 1, 1]

        self.box_size = 0.2
        self.box_density = 0.001
        self.box_null_dist = 0

        self.reward_box_dist = 1.0  # Dense reward for moving the robot towards the box
        self.reward_box_goal = 1.0  # Reward for moving the box towards the goal

        self.hazards_size = 0.3

        self.agent_specific_config()

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

        # Distance from robot to box
        dist_box = self.dist_box()
        gate_dist_box_reward = self.last_dist_box > self.box_null_dist * self.box_size
        reward += (self.last_dist_box - dist_box) * self.reward_box_dist * gate_dist_box_reward
        self.last_dist_box = dist_box

        # Distance from box to goal
        dist_box_goal = self.dist_box_goal()
        reward += (self.last_box_goal - dist_box_goal) * self.reward_box_goal
        self.last_box_goal = dist_box_goal

        if self.goal_achieved:
            reward += self.reward_goal

        return reward

    @property
    def goal_achieved(self):
        return self.dist_box_goal() <= self.goal_size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout"""
        return self.data.body('goal').xpos.copy()

    @property
    def box_pos(self):
        """Helper to get the box position"""
        return self.data.body('box').xpos.copy()

    def agent_specific_config(self):
        if self.robot_base.split('/')[1].split('.')[0] == 'car':
            self.box_size = 0.125  # Box half-radius size
            self.box_keepout = 0.125  # Box keepout radius for placement
            self.box_density = 0.0005

    def specific_reset(self):
        """Reset agent position and set orientation towards desired run
        direction."""
        pass

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards"""
        self.engine.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()

    def build_placements_dict(self):
        """Build a dict of placements.  Happens once during __init__."""
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))

        placements.update(self.placements_dict_from_object('goal'))
        placements.update(self.placements_dict_from_object('box'))
        placements.update(self.placements_dict_from_object('hazard'))
        placements.update(self.placements_dict_from_object('pillar'))

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

        # if self.floor_display_mode:
        #     floor_size = max(self.placements_extents)
        #     world_config['floor_size'] = [floor_size + .1, floor_size + .1, 1]

        # if not self.observe_vision:
        #    world_config['render_context'] = -1  # Hijack this so we don't create context
        # world_config['observe_vision'] = self.observe_vision

        # Extra objects to add to the scene
        world_config['objects'] = {}
        # if self.task_id in ['PushTask0', 'PushTask1', 'PushTask2']:
        world_config['objects']['box'] = get_push_box(
            layout=layout,
            rot=self.random_rot(),
            density=self.box_density,
            size=self.box_size,
        )

        # Extra geoms (immovable objects) to add to the scene
        world_config['geoms'] = {}
        # if self.task_id in ['GoalTask0', 'GoalTask1', 'GoalTask2', 'PushTask0', 'PushTask1', 'PushTask2']:
        world_config['geoms']['goal'] = get_goal(
            layout=layout, rot=self.random_rot(), size=self.goal_size
        )

        return world_config

    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__"""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.build_sensor_observation_space())
        # if self.task == 'push':
        # if self.observe_box_comp:
        # obs_space_dict['box_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
        # if self.observe_box_lidar:
        obs_space_dict['box_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
        )

        # if self.observe_goal_lidar:
        obs_space_dict['goal_lidar'] = gymnasium.spaces.Box(
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
        # if self.task_id in ['PushTask0', 'PushTask1', 'PushTask2']:
        box_pos = self.box_pos
        # if self.observe_box_comp:
        # obs['box_compass'] = self.obs_compass(box_pos)
        # if self.observe_box_lidar:
        obs['box_lidar'] = self.obs_lidar([box_pos], GROUP['box'])

        obs.update(self.get_sensor_obs())

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

    def dist_box(self):
        """Return the distance from the robot to the box (in XY plane only)"""
        return np.sqrt(np.sum(np.square(self.box_pos - self.world.robot_pos())))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position"""
        return np.sqrt(np.sum(np.square(self.box_pos - self.goal_pos)))
