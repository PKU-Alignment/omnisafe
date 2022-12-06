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
"""button"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


@dataclass
class Buttons:
    # Buttons are small immovable spheres, to the environment
    num: int = 0  # Number of buttons to add
    size: float = 0.1
    placements: list = None  # Buttons placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.2  # Buttons keepout radius for placement
    cost: float = 1.0  # Cost for pressing the wrong button, if constrain_buttons
    resampling_delay: float = 10  # Buttons have a timeout period (steps) before resampling

    reward_goal: float = 1.0  # Sparse reward for being inside the goal area
    # Reward is distance towards goal plus a constant for being within range of goal
    # reward_distance should be positive to encourage moving towards the goal
    # if reward_distance is 0, then the reward function is sparse
    reward_distance: float = 1.0  # Dense reward multiplied by the distance moved to the goal

    def get_button(self, index, layout, rot):
        name = f'button{index}'
        geom = {
            'name': name,
            'size': np.ones(3) * self.size,
            'pos': np.r_[layout[name], self.size],
            'rot': rot,
            'type': 'sphere',
            'group': GROUP['button'],
            'rgba': COLOR['button'],
        }
        return geom
