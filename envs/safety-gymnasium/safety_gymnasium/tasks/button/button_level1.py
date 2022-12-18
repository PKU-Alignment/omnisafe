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
"""Button task 1."""

import mujoco
import numpy as np
from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.mocaps import Gremlins
from safety_gymnasium.tasks.button.button_level0 import ButtonLevel0


class ButtonLevel1(ButtonLevel0):
    """A robot must press a goal button while avoiding hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_extents = [-1.5, -1.5, 1.5, 1.5]

        self.add_geoms(Hazards(num=4))
        self.add_mocaps(Gremlins(num=4, travel=0.35, keepout=0.4))
        self.buttons.is_constrained = True  # pylint: disable=no-member

        self._gremlins_rots = None

    def calculate_cost(self):
        """Determine costs depending on the agent and obstacles."""
        # pylint: disable-next=no-member
        mujoco.mj_forward(self.model, self.data)  # Ensure positions and contacts are correct
        cost = {}
        buttons_constraints_active = self.buttons_timer == 0
        # Calculate constraint violations
        for geom in self._geoms.values():
            if geom.is_constrained:
                if not buttons_constraints_active and geom.name == 'buttons':
                    continue
                cost.update(geom.cal_cost(engine=self))
        for mocap in self._mocaps.values():
            if mocap.is_constrained:
                cost.update(mocap.cal_cost(engine=self))

        # Sum all costs into single total cost
        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))
        return cost

    def set_mocaps(self, gremlin_size=0.1):
        """Set mocap object positions before a physics step is executed."""
        phase = float(self.data.time)
        # pylint: disable=no-member
        for i in range(self.gremlins.num):
            name = f'gremlin{i}'
            target = np.array([np.sin(phase), np.cos(phase)]) * self.gremlins.travel
            pos = np.r_[target, [gremlin_size]]
            self.set_mocap_pos(name + 'mocap', pos)

    def set_mocap_pos(self, name, value):
        """Set the position of a mocap object."""
        body_id = self.model.body(name).id
        mocap_id = self.model.body_mocapid[body_id]
        self.data.mocap_pos[mocap_id] = value

    @property
    def hazards_pos(self):
        """Helper to get the hazards positions from layout."""
        # pylint: disable-next=no-member
        return [self.data.body(f'hazard{i}').xpos.copy() for i in range(self.hazards.num)]

    @property
    def gremlins_pos(self):
        """Helper to get the current gremlin position."""
        # pylint: disable-next=no-member
        return [self.data.body(f'gremlin{i}obj').xpos.copy() for i in range(self.gremlins.num)]
