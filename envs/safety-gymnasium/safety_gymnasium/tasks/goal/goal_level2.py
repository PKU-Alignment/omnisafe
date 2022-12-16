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
"""Goal level 2."""

from safety_gymnasium.tasks.goal.goal_level1 import GoalLevel1


class GoalLevel2(GoalLevel1):
    """A robot must navigate to a goal while avoiding more hazards and vases."""

    def __init__(self, config):
        super().__init__(config=config)
        # pylint: disable=no-member

        self.placements_extents = [-2, -2, 2, 2]

        self.hazards.num = 10
        self.vases.num = 10
        self.vases.is_constrained = True
