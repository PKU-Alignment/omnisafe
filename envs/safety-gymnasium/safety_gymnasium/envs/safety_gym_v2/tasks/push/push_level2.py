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
"""push level 2"""

import mujoco
from safety_gymnasium.envs.safety_gym_v2.tasks.push.push_level1 import PushLevel1


class PushLevel2(PushLevel1):
    """A task where agents have to run as fast as possible within a circular zone.

    Rewards are by default shaped.
    """

    def __init__(self, task_config):
        super().__init__(task_config=task_config)

        self.placements_extents = [-2, -2, 2, 2]

        self.hazards.num = 4
        self.pillars.num = 4

    def calculate_cost(self):
        """determine costs depending on agent and obstacles"""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}

        # Conctacts processing
        cost['cost_pillars'] = 0

        for contact in self.data.contact[: self.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.model.geom(g).name for g in geom_ids])
            if any(n.startswith('pillar') for n in geom_names):
                if any(n in self.robot.geom_names for n in geom_names):
                    cost['cost_pillars'] += self.pillars.cost

        # Calculate constraint violations
        cost['cost_hazards'] = 0
        for h_pos in self.hazards_pos:
            h_dist = self.dist_xy(h_pos)
            if h_dist <= self.hazards.size:
                cost['cost_hazards'] += self.hazards.cost * (self.hazards.size - h_dist)

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost
