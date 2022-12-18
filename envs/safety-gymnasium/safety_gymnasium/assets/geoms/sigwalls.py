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

from dataclasses import dataclass

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP


@dataclass
class Sigwalls:
    """Non collision object."""

    name: str = 'sigwalls'
    num: int = 2
    lenth: float = 3.5
    placements: list = None
    locations: tuple = ((1.125, 0), (-1.125, 0))
    keepout: float = 0.

    color: np.array = COLOR['sigwall']
    group: np.array = GROUP['sigwall']
    is_observe_lidar: bool = False
    is_constrained: bool = False

    def get(self, index, layout, rot):
        """To facilitate get specific config for this object."""
        name = f'sigwall{index}'
        geom = {
            'name': name,
            'size': np.array([0.05, self.lenth, 0.3]),
            'pos': np.r_[layout[name], 0.25],
            'rot': 0,
            'type': 'box',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * [1, 1, 1, 0.1]
        }
        return geom
