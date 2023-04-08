# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""OnPolicy Adapter for OmniSafe."""

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.adapter.saute_adapter import SauteAdapter
from omnisafe.utils.config import Config


class SimmerAdapter(SauteAdapter, OnPolicyAdapter):
    """Simmer Adapter for OmniSafe."""

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize the adapter."""
        super(OnPolicyAdapter, self).__init__(env_id, num_envs, seed, cfgs)

        self._safety_budget: torch.Tensor
        self._safety_obs: torch.Tensor

        self._safety_budget = (
            self._cfgs.algo_cfgs.safety_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
        )
        self._upper_budget = (
            self._cfgs.algo_cfgs.upper_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
        )
        self._rel_safety_budget = self._safety_budget / self._upper_budget

        self._ep_budget: torch.Tensor

        assert isinstance(self._env.observation_space, Box), 'Observation space must be Box'
        self._observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space.shape[0] + 1,),
        )

    @property
    def safety_budget(self) -> torch.Tensor:
        """Return the safety budget."""
        return self._safety_budget

    @property
    def upper_budget(self) -> torch.Tensor:
        """Return the upper budget."""
        return self._upper_budget

    @safety_budget.setter
    def safety_budget(self, safety_budget: torch.Tensor) -> None:
        """Set the safety budget."""
        self._safety_budget = safety_budget
        self._rel_safety_budget = self._safety_budget / self._upper_budget
