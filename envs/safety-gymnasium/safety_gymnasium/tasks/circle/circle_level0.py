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
"""Circle level0."""

import numpy as np
from safety_gymnasium.assets.geoms import Circle
from safety_gymnasium.bases.base_task import BaseTask


class CircleLevel0(BaseTask):
    """A agent want to loop around the boundary of circle."""

    def __init__(self, config):
        super().__init__(config=config)

        self.num_steps = 500

        self.agent.placements = [(-0.8, -0.8, 0.8, 0.8)]
        self.agent.keepout = 0

        self.lidar_conf.max_dist = 6

        # Reward for circle goal (complicated formula depending on pos and vel)
        self.reward_factor: float = 1e-1

        self._add_geoms(Circle())

    def calculate_reward(self):
        """The agent should loop around the boundary of circle."""
        reward = 0.0
        # Circle environment reward
        agent_pos = self.agent.pos
        agent_vel = self.agent.vel
        x, y, _ = agent_pos
        u, v, _ = agent_vel
        radius = np.sqrt(x**2 + y**2)
        reward += (
            ((-u * y + v * x) / radius)
            / (1 + np.abs(radius - self.circle.radius))  # pylint: disable=no-member
        ) * self.reward_factor
        return reward

    def specific_reset(self):
        pass

    def specific_step(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Weather the goal of task is achieved."""
        return False
