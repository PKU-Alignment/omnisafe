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
"""group"""

GROUP = {
    # Groups are a mujoco-specific mechanism for selecting which geom objects to "see"
    # We use these for raycasting lidar, where there are different lidar types.
    # These work by turning "on" the group to see and "off" all the other groups.
    # See obs_lidar_natural() for more.
    'goal': 0,
    'push_box': 1,
    'button': 1,
    'wall': 2,
    'pillar': 2,
    'hazard': 3,
    'vase': 4,
    'gremlin': 5,
    'circle': 6,
    'apple': 7,
    'orange': 8,
}
