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
"""Goal level 0."""

from safety_gymnasium.assets.geoms import Goal
from safety_gymnasium.bases import BaseTask


class GoalLevel0(BaseTask):
    """A robot must navigate to a goal."""

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_extents = [-1, -1, 1, 1]

        self.add_geoms(Goal(keepout=0.305))

        self.specific_agent_config()
        self.last_dist_goal = None

    def calculate_reward(self):
        """Determine reward depending on the agent and tasks."""
        # pylint: disable=no-member
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def build_goal(self):
        """Build a new goal position, maybe with resampling due to hazards."""
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Whether the goal of task is achieved."""
        # pylint: disable-next=no-member
        return self.dist_goal() <= self.goal.size

    @property
    def goal_pos(self):
        """Helper to get goal position from layout."""
        return [self.data.body('goal').xpos.copy()]
