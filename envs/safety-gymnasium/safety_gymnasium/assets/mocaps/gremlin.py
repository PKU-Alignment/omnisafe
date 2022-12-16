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
"""Gremlin."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP


@dataclass
class Gremlins:
    """Gremlins (moving objects we should avoid)"""

    name: str = 'gremlins'
    num: int = 0  # Number of gremlins in the world
    size: float = 0.1
    placements: list = None  # Gremlins placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.5  # Radius for keeping out (contains gremlin path)
    travel: float = 0.3  # Radius of the circle traveled in
    contact_cost: float = 1.0  # Cost for touching a gremlin
    dist_threshold: float = 0.2  # Threshold for cost for being too close
    dist_cost: float = 1.0  # Cost for being within distance threshold
    density: float = 0.001

    color: np.array = COLOR['gremlin']
    group: np.array = GROUP['gremlin']
    is_observe_lidar: bool = True
    is_constrained: bool = True

    def get_obj(
        self,
        index,
        layout,
        rot,
    ):
        """To facilitate get specific config for this object"""
        name = f'gremlin{index}obj'
        obj = {
            'name': name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'density': self.density,
            'pos': np.r_[layout[name.replace('obj', '')], self.size],
            'rot': rot,
            'group': self.group,
            'rgba': self.color,
        }
        return obj

    def get_mocap(self, index, layout, rot):
        """To facilitate get specific config for this object"""
        name = f'gremlin{index}mocap'
        mocap = {
            'name': name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'pos': np.r_[layout[name.replace('mocap', '')], self.size],
            'rot': rot,
            'group': self.group,
            'rgba': np.array([1, 1, 1, 0.1]) * self.color,
        }
        return mocap

    def cal_cost(self, engine):
        # Conctacts processing
        cost = {}
        cost['cost_gremlins'] = 0
        for contact in engine.data.contact[: engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('gremlin') for n in geom_names):
                if any(n in engine.robot.geom_names for n in geom_names):
                    # pylint: disable-next=no-member
                    cost['cost_gremlins'] += self.contact_cost

        return cost
