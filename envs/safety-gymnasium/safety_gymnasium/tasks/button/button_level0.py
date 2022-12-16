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
"""Button task 0."""

from collections import OrderedDict

import gymnasium
import mujoco
import numpy as np
from safety_gymnasium.assets.geoms import Buttons
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases import BaseTask


class ButtonLevel0(BaseTask):
    """A robot must press a goal button."""

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_extents = [-1, -1, 1, 1]

        self.add_geoms(Buttons(num=4, is_constrained=False))

        self.specific_agent_config()
        self.last_dist_goal = None
        self.buttons_timer = None
        self.goal_button = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0
        dist_goal = self.dist_goal()
        # pylint: disable-next=no-member
        reward += (self.last_dist_goal - dist_goal) * self.buttons.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.buttons.reward_goal  # pylint: disable=no-member

        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        """Reset the buttons timer."""
        self.buttons_timer = 0

    def specific_step(self):
        """Clock the buttons timer."""
        self.buttons_timer_tick()

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        assert self.buttons.num > 0, 'Must have at least one button.'
        self.build_goal_button()
        self.last_dist_goal = self.dist_goal()
        self.buttons_timer = self.buttons.resampling_delay  # pylint: disable=no-member

    def build_goal_button(self):
        """Pick a new goal button, maybe with resampling due to hazards."""
        # pylint: disable-next=no-member
        self.goal_button = self.random_generator.choice(self.buttons.num)

    def update_world(self):
        pass

    def obs(self):
        """Return the observation of our agent."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Needed to get sensordata correct
        obs = {}

        obs.update(self.obs_sensor())

        for geom in self._geoms.values():
            name = geom.name + '_' + 'lidar'
            if geom.name == 'buttons':
                # Buttons observation is zero while buttons are resetting
                if self.buttons_timer == 0:
                    obs['buttons_lidar'] = self.obs_lidar(self.buttons_pos, geom.group)
                else:
                    obs['buttons_lidar'] = np.zeros(self.lidar_num_bins)
            else:
                obs[name] = self.obs_lidar(getattr(self, geom.name + '_pos'), geom.group)
        for obj in self._objects.values():
            name = obj.name + '_' + 'lidar'
            obs[name] = self.obs_lidar(getattr(self, obj.name + '_pos'), obj.group)
        for mocap in self._mocaps.values():
            name = mocap.name + '_' + 'lidar'
            obs[name] = self.obs_lidar(getattr(self, mocap.name + '_pos'), mocap.group)

        if 'buttons' in self._geoms:
            obs['goal_lidar'] = self.obs_lidar(self.goal_pos, GROUP['goal'])

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

        if 'buttons' in self._geoms:
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
            self.obs_flat_size = sum(np.prod(i.shape) for i in self.obs_space_dict.values())
            self.observation_space = gymnasium.spaces.Box(
                -np.inf, np.inf, (self.obs_flat_size,), dtype=np.float64
            )
        else:
            self.observation_space = gymnasium.spaces.Dict(obs_space_dict)

    def buttons_timer_tick(self):
        """Tick the buttons resampling timer."""
        #  Button timer (used to delay button resampling)
        self.buttons_timer = max(0, self.buttons_timer - 1)

    @property
    def goal_achieved(self):
        """Weather the goal of task is achieved."""
        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if any(n == f'button{self.goal_button}' for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    return True
        return False

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        return [self.data.body(f'button{self.goal_button}').xpos.copy()]

    @property
    def buttons_pos(self):
        """Helper to get the list of button positions."""
        # pylint: disable-next=no-member
        return [self.data.body(f'button{i}').xpos.copy() for i in range(self.buttons.num)]
