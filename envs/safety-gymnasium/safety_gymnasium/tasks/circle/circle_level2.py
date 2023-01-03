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
"""Circle level2."""

from safety_gymnasium.tasks.circle.circle_level1 import CircleLevel1


class CircleLevel2(CircleLevel1):
    """A robot want to loop around the boundary of circle, while avoid going outside the stricter boundaries."""

    def __init__(self, config):
        super().__init__(config=config)

        self.sigwalls.num = 4
