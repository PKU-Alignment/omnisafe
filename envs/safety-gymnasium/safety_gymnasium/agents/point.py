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
"""Point."""

import glfw
import numpy as np
from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator


class Point(BaseAgent):
    """A simple agent constrained to the 2D-plane.

    With one actuator for turning and another for moving forward/backwards.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        random_generator: RandomGenerator,
        placements: list = None,
        locations: list = None,
        keepout: float = 0.4,
        rot: float = None,
    ):
        super().__init__(
            self.__class__.__name__, random_generator, placements, locations, keepout, rot
        )

    def is_alive(self):
        """Point runs until timeout."""
        return True

    def reset(self):
        """No need to reset anything."""

    def debug(self):
        """Apply action which inputted from keyboard."""
        action = np.array([0, 0])
        for key in self.debug_info.keys:
            if key == glfw.KEY_I:
                action[0] += 1
            elif key == glfw.KEY_K:
                action[0] -= 1
            elif key == glfw.KEY_J:
                action[1] += 1
            elif key == glfw.KEY_L:
                action[1] -= 1
        self.apply_action(action)
