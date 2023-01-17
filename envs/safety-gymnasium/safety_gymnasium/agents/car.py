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
"""Car."""

import glfw
import numpy as np
from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator


class Car(BaseAgent):
    """Car is a slightly more complex agent.

    Which has two independently-driven parallel wheels and a free rolling rear wheel.
    Car is not fixed to the 2D-plane, but mostly resides in it.
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
                action += np.array([1, 1])
            elif key == glfw.KEY_K:
                action += np.array([-1, -1])
            elif key == glfw.KEY_J:
                action = np.array([1, -1])
                break
            elif key == glfw.KEY_L:
                action = np.array([-1, 1])
                break
        self.apply_action(action)
