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
"""Push box."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Objects


@dataclass
class PushBox(Objects):  # pylint: disable=too-many-instance-attributes
    """Box parameters (only used if task == 'push')"""

    name: str = 'push_box'
    size: float = 0.2
    placements: list = None  # Box placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.2  # Box keepout radius for placement
    null_dist: float = 2  # Within box_null_dist * box_size radius of box, no box reward given
    density: float = 0.001
    null_dist: float = 0

    reward_box_dist: float = 1.0  # Dense reward for moving the agent towards the box
    reward_box_goal: float = 1.0  # Reward for moving the box towards the goal

    color: np.array = COLOR['push_box']
    group: np.array = GROUP['push_box']
    is_lidar_observed: bool = True
    is_comp_observed: bool = False
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        obj = {
            'name': 'push_box',
            'type': 'box',
            'size': np.ones(3) * self.size,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'density': self.density,
            'group': self.group,
            'rgba': self.color,
        }
        return obj

    def _specific_agent_config(self):
        """Modify the push_box property according to specific agent."""
        if self.agent.__class__.__name__ == 'Car':
            self.size = 0.125  # Box half-radius size
            self.keepout = 0.125  # Box keepout radius for placement
            self.density = 0.0005

    @property
    def pos(self):
        """Helper to get the box position."""
        return self.engine.data.body('push_box').xpos.copy()
