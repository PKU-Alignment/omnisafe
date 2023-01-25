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
"""Push level 0."""

import numpy as np
from safety_gymnasium.assets.free_geoms import PushBox
from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.bases.base_task import BaseTask


class PushLevel0(BaseTask):
    """A agent must push a box to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_conf.extents = [-1, -1, 1, 1]

        self._add_geoms(Goal())
        self._add_free_geoms(PushBox(null_dist=0))

        self.last_dist_box = None
        self.last_box_goal = None
        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        reward = 0.0

        # Distance from agent to box
        dist_box = self.dist_box()
        # pylint: disable-next=no-member
        gate_dist_box_reward = self.last_dist_box > self.push_box.null_dist * self.push_box.size
        reward += (
            # pylint: disable-next=no-member
            (self.last_dist_box - dist_box)
            * self.push_box.reward_box_dist  # pylint: disable=no-member
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

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()
        self.last_dist_box = self.dist_box()
        self.last_box_goal = self.dist_box_goal()

    def dist_box(self):
        """Return the distance. from the agent to the box (in XY plane only)"""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.agent.pos)))

    def dist_box_goal(self):
        """Return the distance from the box to the goal XY position."""
        # pylint: disable-next=no-member
        return np.sqrt(np.sum(np.square(self.push_box.pos - self.goal.pos)))

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_box_goal() <= self.goal.size
