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
"""Wrapper to enforce the proper ordering of environment operations."""
# pylint: disable=all

import gymnasium
from gymnasium.error import ResetNeeded


class OrderEnforcing(gymnasium.Wrapper):
    """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

    Example:
        >>> from gymnasium.envs.classic_control import CartPoleEnv
        >>> env = CartPoleEnv()
        >>> env = OrderEnforcing(env)
        >>> env.step(0)
        ResetNeeded: Cannot call env.step() before calling env.reset()
        >>> env.render()
        ResetNeeded: Cannot call env.render() before calling env.reset()
        >>> env.reset()
        >>> env.render()
        >>> env.step(0)
    """

    def __init__(self, env: gymnasium.Env, disable_render_order_enforcing: bool = False):
        """A wrapper that will produce an error if :meth:`step` is called before an initial :meth:`reset`.

        Args:
            env: The environment to wrap
            disable_render_order_enforcing: If to disable render order enforcing
        """
        super().__init__(env)
        self._has_reset: bool = False
        self._disable_render_order_enforcing: bool = disable_render_order_enforcing

    def step(self, action):
        """Steps through the environment with `kwargs`."""
        if not self._has_reset:
            raise ResetNeeded('Cannot call env.step() before calling env.reset()')
        return self.env.step(action)

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        self._has_reset = True
        return self.env.reset(**kwargs)

    def render(self, *args, **kwargs):
        """Renders the environment with `kwargs`."""
        if not self._disable_render_order_enforcing and not self._has_reset:
            raise ResetNeeded(
                'Cannot call `env.render()` before calling `env.reset()`, if this is a intended action, '
                'set `disable_render_order_enforcing=True` on the OrderEnforcer wrapper.'
            )
        return self.env.render(*args, **kwargs)

    @property
    def has_reset(self):
        """Returns if the environment has been reset before."""
        return self._has_reset
