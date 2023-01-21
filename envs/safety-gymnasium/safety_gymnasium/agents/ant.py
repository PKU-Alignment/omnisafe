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
"""Ant."""

import numpy as np
from safety_gymnasium.bases.base_agent import BaseAgent
from safety_gymnasium.utils.random_generator import RandomGenerator


class Ant(BaseAgent):
    """The ant is a quadrupedal agent composed of nine rigid links,

    including a torso and four legs. Each leg consists of two actuators
    which are controlled based on torques.
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
        """Die if it touches the ground."""
        return self.engine.data.body('agent').xpos.copy()[2] > 0.08

    def reset(self):
        """Improved spawning behavior of Ant agent.

        Ankle joints are set to 90° position which enables uniform random
        policies better exploration and occasionally generates forward
        movements.
        """
        for i in range(self.body_info.nu):
            noise = self.random_generator.uniform(low=-0.1, high=0.1)
            pos = noise if i % 2 == 0 else np.pi / 2 + noise
            joint_id = self.engine.model.actuator(i).trnid
            if i in (3, 5):
                pos *= -1
            self.engine.data.joint(joint_id[0]).qpos = pos
