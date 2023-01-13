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
"""Passive environment checker."""

from gymnasium.core import ActType
from gymnasium.wrappers.env_checker import PassiveEnvChecker
from safety_gymnasium.utils.passive_env_checker import env_step_passive_checker


class SafePassiveEnvChecker(PassiveEnvChecker):
    """Passive environment checker.

    A passive environment checker wrapper for an environment's observation and action space
    along with the reset, step and render functions.
    """

    def step(self, action: ActType):
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)

        return self.env.step(action)
