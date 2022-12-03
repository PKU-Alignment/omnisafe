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
"""goal"""

import numpy as np
from safety_gymnasium.envs.safety_gym_v2.assets.color import COLOR
from safety_gymnasium.envs.safety_gym_v2.assets.group import GROUP


# =========================================================================================================
# Extra geoms (immovable objects) to add to the scene
def get_goal(layout, rot, size=0.3):
    geom = {
        'name': 'goal',
        'size': [size, size / 2],
        'pos': np.r_[layout['goal'], size / 2 + 1e-2],
        'rot': rot,
        'type': 'cylinder',
        'contype': 0,
        'conaffinity': 0,
        'group': GROUP['goal'],
        'rgba': COLOR['goal'] * [1, 1, 1, 0.25],
    }  # transparent
    return geom
