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
"""pillar"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.assets.color import COLOR
from safety_gymnasium.envs.assets.group import GROUP


@dataclass
class Pillars:
    """Pillars (immovable obstacles we should not touch)"""

    name: str = 'pillars'
    num: int = 0  # Number of pillars in the world
    placements: list = None  # Pillars placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.3  # Radius for placement of pillars
    cost: float = 1.0  # Cost (per step) for being in contact with a pillar

    color: np.array = COLOR['pillar']
    group: np.array = GROUP['pillar']
    is_observe_lidar: bool = True

    # pylint: disable-next=too-many-arguments
    def get(self, index, layout, rot, size=0.2, height=0.5):
        """To facilitate get specific config for this object."""
        name = f'pillar{index}'
        geom = {
            'name': name,
            'size': [size, height],
            'pos': np.r_[layout[name], height],
            'rot': rot,
            'type': 'cylinder',
            'group': self.group,
            'rgba': self.color,
        }
        return geom
