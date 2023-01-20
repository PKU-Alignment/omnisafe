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
from safety_gymnasium.bases.base_obstacle import Mocaps


@dataclass
class Gremlins(Mocaps):  # pylint: disable=too-many-instance-attributes
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
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object"""
        return {'obj': self.get_obj(xy_pos, rot), 'mocap': self.get_mocap(xy_pos, rot)}

    def get_obj(self, xy_pos, rot):
        """To facilitate get objects config for this object"""
        obj = {
            'name': self.name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'density': self.density,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'group': self.group,
            'rgba': self.color,
        }
        return obj

    def get_mocap(self, xy_pos, rot):
        """To facilitate get mocaps config for this object"""
        mocap = {
            'name': self.name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'group': self.group,
            'rgba': np.array([1, 1, 1, 0.1]) * self.color,
        }
        return mocap

    def cal_cost(self):
        """Contacts processing."""
        cost = {}
        if not self.is_constrained:
            return cost
        cost['cost_gremlins'] = 0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('gremlin') for n in geom_names):
                if any(n in self.agent.body_info.geom_names for n in geom_names):
                    # pylint: disable-next=no-member
                    cost['cost_gremlins'] += self.contact_cost

        return cost

    def move(self):
        """Set mocap object positions before a physics step is executed."""
        phase = float(self.engine.data.time)
        for i in range(self.num):
            name = f'gremlin{i}'
            target = np.array([np.sin(phase), np.cos(phase)]) * self.travel
            pos = np.r_[target, [self.size]]
            self.set_mocap_pos(name + 'mocap', pos)

    @property
    def pos(self):
        """Helper to get the current gremlin position."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'gremlin{i}obj').xpos.copy() for i in range(self.num)]
