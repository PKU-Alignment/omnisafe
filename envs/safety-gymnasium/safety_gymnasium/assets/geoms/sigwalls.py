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
class Sigwalls:  # pylint: disable=too-many-instance-attributes
    """Non collision object."""

    name: str = 'sigwalls'
    num: int = 2
    locate_factor: float = 1.125
    size: float = 3.5
    placements: list = None
    keepout: float = 0.0

    color: np.array = COLOR['sigwall']
    group: np.array = GROUP['sigwall']
    is_observe_lidar: bool = False
    is_constrained: bool = False

    def __post_init__(self):
        assert self.num == 2 or self.num == 4, "Sigwalls are specific for Circle and Run tasks."
        assert self.locate_factor >= 0, "For cost calculation, the locate_factor\
                                         must be greater than or equal to zero."
        self.locations = [(self.locate_factor, 0), (-self.locate_factor, 0),
                        (0, self.locate_factor), (0, -self.locate_factor)]

    def get(self, index, layout, rot):  # pylint: disable=unused-argument
        """To facilitate get specific config for this object."""
        name = f'sigwall{index}'
        geom = {
            'name': name,
            'size': np.array([0.05, self.size, 0.3]),
            'pos': np.r_[layout[name], 0.25],
            'rot': 0,
            'type': 'box',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * [1, 1, 1, 0.1],
        }
        if index >= 2:
            geom.update({
            'rot': np.pi / 2,
            })
        return geom

    def cal_cost(self, engine):
        """Contacts Processing."""
        cost = {}
        cost['cost_out_of_boundary'] = np.abs(engine.robot_pos[0]) > self.locate_factor
        if self.num == 4:
            cost['cost_out_of_boundary'] = (cost['cost_out_of_boundary'] or
                                            np.abs(engine.robot_pos[1]) > self.locate_factor)

        return cost
