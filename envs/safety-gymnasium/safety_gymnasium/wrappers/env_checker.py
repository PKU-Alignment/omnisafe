# The MIT License

# Copyright (c) 2016 OpenAI
# Copyright (c) 2022 Farama Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================

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
"""A passive environment checker wrapper for an environment's observation and action space along with the reset, step and render functions."""
# pylint: disable=all

import gymnasium
from gymnasium.core import ActType
from safety_gymnasium.utils.passive_env_checker import (
    check_action_space,
    check_observation_space,
    env_render_passive_checker,
    env_reset_passive_checker,
    env_step_passive_checker,
)


class PassiveEnvChecker(gymnasium.Wrapper):
    """A passive environment checker wrapper that surrounds the step, reset and render functions to check they follow the gymnasium API."""

    def __init__(self, env):
        """Initialises the wrapper with the environments, run the observation and action space tests."""
        super().__init__(env)

        assert hasattr(
            env, 'action_space'
        ), 'The environment must specify an action space. https://gymnasium.farama.org/content/environment_creation/'
        check_action_space(env.action_space)
        assert hasattr(
            env, 'observation_space'
        ), 'The environment must specify an observation space. https://gymnasium.farama.org/content/environment_creation/'
        check_observation_space(env.observation_space)

        self.checked_reset = False
        self.checked_step = False
        self.checked_render = False

    def step(self, action: ActType):
        """Steps through the environment that on the first call will run the `passive_env_step_check`."""
        if self.checked_step is False:
            self.checked_step = True
            return env_step_passive_checker(self.env, action)
        else:
            return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment that on the first call will run the `passive_env_reset_check`."""
        if self.checked_reset is False:
            self.checked_reset = True
            return env_reset_passive_checker(self.env, **kwargs)
        else:
            return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment that on the first call will run the `passive_env_render_check`."""
        if self.checked_render is False:
            self.checked_render = True
            return env_render_passive_checker(self.env, *args, **kwargs)
        else:
            return self.env.render(*args, **kwargs)
