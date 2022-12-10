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
"""apple"""

from dataclasses import dataclass, field

import numpy as np
from safety_gymnasium.envs.assets.color import COLOR
from safety_gymnasium.envs.assets.group import GROUP


@dataclass
class Apples:
    """Apples and Oranges are as same as Goal.

    While they can be instantiated more than one.
    And one can define different rewards for Apple and Orange.
    """

    name: str = 'apples'
    num: int = 0
    placements: list = None
    locations: list = field(default_factory=list)
    keepout: float = 0.3

    color: np.array = COLOR['apple']
    group: np.array = GROUP['apple']
    is_observe_lidar: bool = True
    is_observe_comp: bool = False
