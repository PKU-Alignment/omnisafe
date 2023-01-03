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
"""Goal."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP


@dataclass
class Goal:  # pylint: disable=too-many-instance-attributes
    """Goal parameters."""

    name: str = 'goal'
    size: float = 0.3
    placements: list = None  # Placements where goal may appear (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.4  # Keepout radius when placing goals

    reward_goal: float = 1.0  # Sparse reward for being inside the goal area
    # Reward is distance towards goal plus a constant for being within range of goal
    # reward_distance should be positive to encourage moving towards the goal
    # if reward_distance is 0, then the reward function is sparse
    reward_distance: float = 1.0  # Dense reward multiplied by the distance moved to the goal

    color: np.array = COLOR['goal']
    group: np.array = GROUP['goal']
    is_observe_lidar: bool = True
    is_observe_comp: bool = False
    is_constrained: bool = False
    is_meshed: bool = False

    def get(self, layout, rot):
        """To facilitate get specific config for this object."""
        geom = {
            'name': 'goal',
            'size': [self.size, self.size / 2],
            'pos': np.r_[layout['goal'], self.size / 2 + 1e-2],
            'rot': rot,
            'type': 'cylinder',
            'contype': 0,
            'conaffinity': 0,
            'group': self.group,
            'rgba': self.color * [1, 1, 1, 0.25],  # transparent
        }
        if self.is_meshed:
            geom.update(
                {
                    'type': 'mesh',
                    'mesh': 'flower_bush',
                    'material': 'flower_bush',
                    'euler': [np.pi / 2, 0, 0],
                }
            )
        return geom
