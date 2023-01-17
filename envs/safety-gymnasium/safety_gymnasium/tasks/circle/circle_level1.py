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
"""Circle level1."""

from safety_gymnasium.assets.geoms import Sigwalls
from safety_gymnasium.tasks.circle.circle_level0 import CircleLevel0


class CircleLevel1(CircleLevel0):
    """A agent want to loop around the boundary of circle, while avoid going outside the boundaries."""

    def __init__(self, config):
        super().__init__(config=config)

        self._add_geoms(Sigwalls(num=2, locate_factor=1.125, is_constrained=True))
