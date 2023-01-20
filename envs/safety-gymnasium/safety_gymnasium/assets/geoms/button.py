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
"""Button."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Geoms


@dataclass
class Buttons(Geoms):  # pylint: disable=too-many-instance-attributes
    """Buttons are small immovable spheres, to the environment."""

    name: str = 'buttons'
    num: int = 0  # Number of buttons to add
    size: float = 0.1
    placements: list = None  # Buttons placements list (defaults to full extents)
    locations: list = field(default_factory=list)  # Fixed locations to override placements
    keepout: float = 0.2  # Buttons keepout radius for placement
    goal_button: int = None  # Button to be the goal

    resampling_delay: float = 10  # Buttons have a timeout period (steps) before resampling
    timer: int = None

    cost: float = 1.0  # Cost for pressing the wrong button, if constrain_buttons
    reward_goal: float = 1.0  # Sparse reward for being inside the goal area
    # Reward is distance towards goal plus a constant for being within range of goal
    # reward_distance should be positive to encourage moving towards the goal
    # if reward_distance is 0, then the reward function is sparse
    reward_distance: float = 1.0  # Dense reward multiplied by the distance moved to the goal

    color: np.array = COLOR['button']
    group: np.array = GROUP['button']
    is_lidar_observed: bool = True
    is_constrained: bool = True

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""
        geom = {
            'name': self.name,
            'size': np.ones(3) * self.size,
            'pos': np.r_[xy_pos, self.size],
            'rot': rot,
            'type': 'sphere',
            'group': self.group,
            'rgba': self.color,
        }
        return geom

    def cal_cost(self):
        """Contacts processing."""
        assert (
            self.goal_button is not None
        ), 'Please make sure this method can get infomation about goal button.'
        cost = {}
        buttons_constraints_active = self.timer == 0
        if not self.is_constrained or not buttons_constraints_active:
            return cost
        cost['cost_buttons'] = 0
        for contact in self.engine.data.contact[: self.engine.data.ncon]:
            geom_ids = [contact.geom1, contact.geom2]
            geom_names = sorted([self.engine.model.geom(g).name for g in geom_ids])
            if any(n.startswith('button') for n in geom_names):
                if any(n in self.agent.body_info.geom_names for n in geom_names):
                    if not any(n == f'button{self.goal_button}' for n in geom_names):
                        # pylint: disable-next=no-member
                        cost['cost_buttons'] += self.cost
        return cost

    def timer_tick(self):
        """Tick the buttons resampling timer."""
        #  Button timer (used to delay button resampling)
        self.timer = max(0, self.timer - 1)

    def reset_timer(self):
        """Reset the timer to the resampling delay."""
        self.timer = self.resampling_delay

    @property
    def pos(self):
        """Helper to get the list of button positions."""
        # pylint: disable-next=no-member
        return [self.engine.data.body(f'button{i}').xpos.copy() for i in range(self.num)]
