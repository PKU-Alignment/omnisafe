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
"""Goal level 1."""

from safety_gymnasium.assets.geoms import Hazards
from safety_gymnasium.assets.objects import Vases
from safety_gymnasium.tasks.goal.goal_level0 import GoalLevel0


class GoalLevel1(GoalLevel0):
    """A agent must navigate to a goal while avoiding hazards.

    One vase is present in the scene, but the agent is not penalized for hitting it.
    """

    def __init__(self, config):
        super().__init__(config=config)

        self.placements_conf.extents = [-1.5, -1.5, 1.5, 1.5]

        self._add_geoms(Hazards(num=8, keepout=0.18))
        self._add_objects(Vases(num=1, is_constrained=False))
