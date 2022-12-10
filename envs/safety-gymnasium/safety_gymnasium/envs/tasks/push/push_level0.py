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
"""push level 0"""

import mujoco
import numpy as np
from safety_gymnasium.envs.assets.geoms import Goal
from safety_gymnasium.envs.assets.objects import PushBox
from safety_gymnasium.envs.bases import BaseTask


class PushLevel0(BaseTask):
    """A robot must push a box to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_extents = [-1, -1, 1, 1]

        self.add_geoms(Goal())
        self.add_objects(PushBox())

        self.specific_agent_config()
        self.last_dist_box = None
        self.last_box_goal = None
        self.last_dist_goal = None

    def calculate_cost(self):
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0

        # Distance from robot to box
        dist_box = self.dist_box()
        # pylint: disable-next=no-member
        gate_dist_box_reward = self.last_dist_box > self.push_box.null_dist * self.push_box.size
        reward += (
            # pylint: disable-next=no-member
            (self.last_dist_box - dist_box)
            * self.push_box.reward_box_dist
            * gate_dist_box_reward
        )
        self.last_dist_box = dist_box

        # Distance from box to goal
        dist_box_goal = self.dist_box_goal()
        # pylint: disable-next=no-member
        reward += (self.last_box_goal - dist_box_goal) * self.push_box.reward_box_goal
        self.last_box_goal = dist_box_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal  # pylint: disable=no-member

        return reward

    def specific_agent_config(self):
        if self.robot.base.split('/')[1].split('.')[0] == 'car':
            # pylint: disable=no-member
            self.push_box.size = 0.125  # Box half-radius size
            self.push_box.keepout = 0.125  # Box keepout radius for placement
            self.push_box.density = 0.0005

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()

    def update_world(self):
        pass

    def dist_box(self):
        """Return the distance. from the robot to the box (in XY plane only)"""
        return np.sqrt(np.sum(np.square(self.push_box_pos[0] - self.world.robot_pos())))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position."""
        return np.sqrt(np.sum(np.square(self.push_box_pos[0] - self.goal_pos[0])))

    @property
    def goal_achieved(self):
        """Weather the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_box_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        return [self.data.body('goal').xpos.copy()]

    @property
    def push_box_pos(self):
        """Helper to get the box position."""
        return [self.data.body('push_box').xpos.copy()]
