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
"""Pillar."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Geom


@dataclass
class Pillars(Geom):  # pylint: disable=too-many-instance-attributes
    """Pillars (immovable obstacles we should not touch)"""

    name: str = 'pillars'
    num: int = 0  # Number of pillars in the world
    size: float = 0.2  # Size of pillars
    height: float = 0.5  # Height of pillars
    placements: list = None  # Pillars placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.3  # Radius for placement of pillars
    cost: float = 1.0  # Cost (per step) for being in contact with a pillar

    color: np.array = COLOR['pillar']
    group: np.array = GROUP['pillar']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    # pylint: disable-next=too-many-arguments
    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        geom = {
            'name': self.name,
            'size': [self.size, self.height],
            'pos': np.r_[xy_pos, self.height],
            'rot': rot,
            'type': 'cylinder',
            'group': self.group,
            'rgba': self.color,
        }
        return geom

    def cal_cost(self):
        """Contacts processing."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_pillars'] = 0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('pillar') for n in geom_names):
                if any(n in self.agent.body_info.geom_names for n in geom_names):
                    # pylint: disable-next=no-member
                    cost['cost_pillars'] += self.cost

        return cost

    @property
    def pos(self):
        """Helper to get list of pillar positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'pillar{i}').xpos.copy() for i in range(self.num)]
