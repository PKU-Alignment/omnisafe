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
"""Early terminated wrapper."""

from typing import Dict, Tuple, TypeVar

import numpy as np

from omnisafe.utils.tools import expand_dims
from omnisafe.wrappers.cmdp_wrapper import CMDPWrapper
from omnisafe.wrappers.wrapper_registry import WRAPPER_REGISTRY


RenderFrame = TypeVar('RenderFrame')


@WRAPPER_REGISTRY.register
class EarlyTerminatedWrapper(CMDPWrapper):  # pylint: disable=too-many-instance-attributes
    """Implementation of the environment wrapper for early-terminated algorithms.

    ``omnisafe`` use different environment wrappers for different kinds of algorithms.
    This is the environment wrapper for early-terminated algorithms.

    .. note::
        The only difference between this wrapper and :class:`OnPolicyEnvWrapper` is that,
        this wrapper terminates the episode when the cost is unequal to 0.
        Any on-policy algorithm can use this wrapper,
        to convert itself into an early-terminated algorithm.
        ``omnisafe`` provides a implementation of :class:`PPOEarlyTerminated`,
        and :class:`PPOLagarlyTerminated`.
    """

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict]:
        """Step the environment.

        The environment will be stepped by the action from the agent.
        Corresponding to the Markov Decision Process,
        the environment will return the ``next observation``,
        ``reward``, ``cost``, ``terminated``, ``truncated`` and ``info``.

        Args:
            action (np.ndarray): action.
        """
        next_obs, reward, cost, terminated, truncated, info = self.env.step(action.squeeze())
        if self.cfgs.num_envs == 1:
            next_obs, reward, cost, terminated, truncated, info = expand_dims(
                next_obs, reward, cost, terminated, truncated, info
            )
            if terminated | truncated:
                next_obs, info = self.reset()
        for idx, single_cost in enumerate(cost):
            if single_cost:
                terminated[idx] = True
        self.rollout_data.rollout_log.ep_ret += reward
        self.rollout_data.rollout_log.ep_costs += cost
        self.rollout_data.rollout_log.ep_len += np.ones(self.cfgs.num_envs)
        return next_obs, reward, cost, terminated, truncated, info
