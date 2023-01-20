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
"""Wall."""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.assets.color import COLOR
from safety_gymnasium.assets.group import GROUP
from safety_gymnasium.bases.base_obstacle import Geoms


@dataclass
class Walls(Geoms):  # pylint: disable=too-many-instance-attributes
    """Walls - barriers in the environment not associated with any constraint.

    # NOTE: this is probably best to be auto-generated than manually specified.
    """

    name: str = 'walls'
    num: int = 0  # Number of walls
    placements: list = None  # This should not be used
    locations: list = field(default_factory=list)  # This should be used and length == walls_num
    keepout: float = 0.0  # This should not be used

    color: np.array = COLOR['wall']
    group: np.array = GROUP['wall']
    is_lidar_observed: bool = True
    is_constrained: bool = False

    def get_config(self, xy_pos, rot):
        """To facilitate get specific config for this object."""

    @property
    def pos(self):
        """Helper to get list of Sigwalls positions."""
