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
"""color"""

import numpy as np


COLOR = {
    # Distinct colors for different types of objects.
    # For now this is mostly used for visualization.
    # This also affects the vision observation, so if training from pixels.
    'push_box': np.array([1, 1, 0, 1]),
    'button': np.array([1, 0.5, 0, 1]),
    'goal': np.array([0, 1, 0, 1]),
    'vase': np.array([0, 1, 1, 1]),
    'hazard': np.array([0, 0, 1, 1]),
    'pillar': np.array([0.5, 0.5, 1, 1]),
    'wall': np.array([0.5, 0.5, 0.5, 1]),
    'gremlin': np.array([0.5, 0, 1, 1]),
    'circle': np.array([0, 1, 0, 1]),
    'red': np.array([1, 0, 0, 1]),
    'apple': np.array([0.835, 0.169, 0.169, 1]),
    'orange': np.array([1, 0.6, 0, 1]),
}
