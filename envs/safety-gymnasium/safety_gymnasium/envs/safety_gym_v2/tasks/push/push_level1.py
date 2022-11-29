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
"""push level 1"""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.goal import get_goal
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP
from safety_gymnasium.envs.safety_gym_v2.assets.hazard import get_hazard
from safety_gymnasium.envs.safety_gym_v2.assets.pillar import get_pillar
from safety_gymnasium.envs.safety_gym_v2.assets.push_box import get_push_box
from safety_gymnasium.envs.safety_gym_v2.tasks.push.push_level0 import PushLevel0


class PushLevel1(PushLevel0):
    """A task where agents have to run as fast as possible within a circular zone.

    Rewards are by default shaped.
    """

    def __init__(self, task_config):
        super().__init__(task_config=task_config)

        self.placements_extents = [-1.5, -1.5, 1.5, 1.5]
        self.hazards_num = 2
        self.pillars_num = 1

    def calculate_cost(self, **kwargs):
        """determine costs depending on agent and obstacles"""
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Calculate constraint violations
        cost['cost_hazards'] = 0
        for h_pos in self.hazards_pos:
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.hazards_size:
                cost['cost_hazards'] += self.hazards_cost * (self.hazards_size - h_dist)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost

    @property
    def pillars_pos(self):
        """Helper to get list of pillar positions"""
        return [self.data.body(f'pillar{i}').xpos.copy() for i in range(self.pillars_num)]

    @property
    def hazards_pos(self):
        """Helper to get the hazards positions from layout"""
        return [self.data.body(f'hazard{i}').xpos.copy() for i in range(self.hazards_num)]

    def build_placements_dict(self):
        """Build a dict of placements.  Happens once during __init__."""
        # Dictionary is map from object name -> tuple of (placements list, keepout)
        placements = {}

        placements.update(self.placements_dict_from_object('robot'))
        placements.update(self.placements_dict_from_object('wall'))

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

        # if self.hazards_num:
        for i in range(self.hazards_num):
            name = f'hazard{i}'
            world_config['geoms'][name] = get_hazard(
                index=i, layout=layout, rot=self.random_rot(), size=self.hazards_size
            )
        # if self.pillars_num:
        for i in range(self.pillars_num):
            name = f'pillar{i}'
            world_config['geoms'][name] = get_pillar(index=i, layout=layout, rot=self.random_rot())

        return world_config

    def build_observation_space(self):
        """Construct observtion space.  Happens only once at during __init__"""
        obs_space_dict = OrderedDict()  # See self.obs()

        obs_space_dict.update(self.build_sensor_observation_space())

        # if self.task == 'push':
        # if self.observe_box_comp:
        #     obs_space_dict['box_compass'] = gym.spaces.Box(-1.0, 1.0, (self.compass_shape,), dtype=np.float32)
        # if self.observe_box_lidar:
        obs_space_dict['box_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
        )

        # if self.observe_goal_lidar:
        obs_space_dict['goal_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
        )

        # if self.observe_hazards:
        obs_space_dict['hazards_lidar'] = gymnasium.spaces.Box(
            0.0, 1.0, (self.lidar_num_bins,), dtype=np.float64
        )

        # if self.pillars_num and self.observe_pillars:
        obs_space_dict['pillars_lidar'] = gymnasium.spaces.Box(
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
        #     obs['box_compass'] = self.obs_compass(box_pos)
        # if self.observe_box_lidar:
        obs['box_lidar'] = self.obs_lidar([box_pos], GROUP['box'])

        obs.update(self.get_sensor_obs())

        # if self.observe_hazards:
        obs['hazards_lidar'] = self.obs_lidar(self.hazards_pos, GROUP['hazard'])

        # if self.pillars_num and self.observe_pillars:
        obs['pillars_lidar'] = self.obs_lidar(self.pillars_pos, GROUP['pillar'])

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
