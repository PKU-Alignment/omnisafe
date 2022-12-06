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
"""gremlin"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


@dataclass
class Gremlins:
    # Gremlins (moving objects we should avoid)
    num: int = 0  # Number of gremlins in the world
    placements: list = None  # Gremlins placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.5  # Radius for keeping out (contains gremlin path)
    travel: float = 0.3  # Radius of the circle traveled in
    contact_cost: float = 1.0  # Cost for touching a gremlin
    dist_threshold: float = 0.2  # Threshold for cost for being too close
    dist_cost: float = 1.0  # Cost for being within distance threshold
    density: float = (0.001,)
    size: float = (0.1,)

    def get_gremlin(
        self,
        index,
        layout,
        rot,
    ):
        name = f'gremlin{index}obj'
        object = {
            'name': name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'density': self.density,
            'pos': np.r_[layout[name.replace('obj', '')], self.size],
            'rot': rot,
            'group': GROUP['gremlin'],
            'rgba': COLOR['gremlin'],
        }
        return object

    def get_mocap_gremlin(self, index, layout, rot):
        name = f'gremlin{index}mocap'
        mocap = {
            'name': name,
            'size': np.ones(3) * self.size,
            'type': 'box',
            'pos': np.r_[layout[name.replace('mocap', '')], self.size],
            'rot': rot,
            'group': GROUP['gremlin'],
            'rgba': np.array([1, 1, 1, 0.1]) * COLOR['gremlin'],
        }
        return mocap
