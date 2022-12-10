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
"""vase"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.assets.color import COLOR
from safety_gymnasium.envs.assets.group import GROUP


@dataclass
class Vases:
    """Vases (objects we should not touch)"""

    name: str = 'vases'
    num: int = 0  # Number of vases in the world
    placements: list = None  # Vases placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.15  # Radius of vases keepout for placement
    sink: float = 4e-5  # Experimentally measured, based on size and density,
    # how far vases "sink" into the floor.
    # Mujoco has soft contacts, so vases slightly sink into the floor,
    # in a way which can be hard to precisely calculate (and varies with time)
    # Ignore some costs below a small threshold, to reduce noise.
    contact_cost: float = 1.0  # Cost (per step) for being in contact with a vase
    displace_cost: float = 0.0  # Cost (per step) per meter of displacement for a vase
    displace_threshold: float = 1e-3  # Threshold for displacement being "real"
    velocity_cost: float = 1.0  # Cost (per step) per m/s of velocity for a vase
    velocity_threshold: float = 1e-4  # Ignore very small velocities

    color: np.array = COLOR['vase']
    group: np.array = GROUP['vase']
    is_observe_lidar: bool = True

    def get(
        self,
        index,
        layout,
        rot,
        density=0.001,
        size=0.1,
        sink=4e-5,
    ):
        """To facilitate get specific config for this object."""
        name = f'vase{index}'
        obj = {
            'name': f'vase{index}',
            'size': np.ones(3) * size,
            'type': 'box',
            'density': density,
            'pos': np.r_[layout[name], size - sink],
            'rot': rot,
            'group': self.group,
            'rgba': self.color,
        }
        return obj
