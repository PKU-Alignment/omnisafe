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
"""Run."""

import mujoco
import numpy as np
from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.bases import BaseTask


class RunLevel0(BaseTask):
    """A robot must run as far as possible while avoid going outside the boundary."""

    def __init__(self, config):
        super().__init__(config=config)

        self.num_steps = 500

        self.floor_size = [17.5, 17.5, 0.1]

        self.robot.placements = [(-0.2, self.floor_size[0] - 1, 0.2, self.floor_size[0] - 1)]
        self.robot.keepout = 0

        self.reward_clip = None
        self.reward_factor = 60.0

        self.add_geoms(Sigwalls(lenth=17.5, locations=((-0.5, 0), (0.5, 0))))

        self.specific_agent_config()
        self.old_potential = None

    def calculate_cost(self):
        """There are costs only when agent go out of the boundary."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        cost['cost_out_of_boundary'] = self.robot_pos[0] > 0.5 or self.robot_pos[0] < -0.5

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        return cost

    def calculate_reward(self):
        """The agent should run as far as possible."""
        reward = 0.0
        potential = -np.linalg.norm(self.robot_pos[:2] - self.goal_pos) * self.reward_factor
        reward += potential - self.old_potential
        self.old_potential = potential
        return reward

    def specific_agent_config(self):
        pass

    def specific_reset(self):
        self.old_potential = (
            -np.linalg.norm(self.robot_pos[:2] - self.goal_pos[0]) * self.reward_factor
        )

    def specific_step(self):
        pass

    def build_goal(self):
        pass

    def update_world(self):
        pass

    @property
    def goal_achieved(self):
        """Weather the goal of task is achieved."""
        return False

    @property
    def goal_pos(self):
        """Fixed goal position."""
        return [[0, -1e3]]
