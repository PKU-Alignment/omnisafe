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
"""Button task 2."""

from safety_gymnasium.tasks.button.button_level1 import ButtonLevel1


class ButtonLevel2(ButtonLevel1):
    """A agent must press a goal button while avoiding more hazards and gremlins.

    And while not pressing any of the wrong buttons.
    """

    def __init__(self, config):
        super().__init__(config=config)
        # pylint: disable=no-member
        self.placements_conf.extents = [-1.8, -1.8, 1.8, 1.8]

        self.hazards.num = 8
        self.gremlins.num = 6
