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
"""Hazard."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP


@dataclass
class Hazards:  # pylint: disable=too-many-instance-attributes
    """Hazardous areas."""

    name: str = 'hazards'
    num: int = 0  # Number of hazards in an environment
    size: float = 0.2
    placements: list = None  # Placements list for hazards (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.4  # Radius of hazard keepout for placement
    keepout: float = 0.18
    cost: float = 1.0  # Cost (per step) for violating the constraint

    color: np.array = COLOR['hazard']
    group: np.array = GROUP['hazard']
    is_observe_lidar: bool = True
    is_constrained: bool = True

    def get(self, index, layout, rot):
        """To facilitate get specific config for this object."""
        name = f'hazard{index}'
        geom = {
            'name': name,
            'size': [self.size, 1e-2],  # self.hazards_size / 2],
            'pos': np.r_[layout[name], 2e-2],  # self.hazards_size / 2 + 1e-2],
            'rot': rot,
            'type': 'cylinder',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * [1, 1, 1, 0.25],  # transparent
        }
        return geom

    def cal_cost(self, engine):
        """Contacts Processing."""
        cost = {}
        cost['cost_hazards'] = 0
        for h_pos in engine.hazards_pos:
            h_dist = engine.dist_xy(h_pos)
            # pylint: disable=no-member
            if h_dist <= self.size:
                cost['cost_hazards'] += self.cost * (self.size - h_dist)

        return cost
