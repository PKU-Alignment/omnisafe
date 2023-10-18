# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Simmer Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.adapter.saute_adapter import SauteAdapter
from omnisafe.common.simmer_agent import BaseSimmerAgent, SimmerPIDAgent
from omnisafe.utils.config import Config


class SimmerAdapter(SauteAdapter):
    """Simmer Adapter for OmniSafe.

    Simmer is a safe RL algorithm that uses a safety budget to control the exploration of the RL
    agent. Similar to :class:`SauteEnvWrapper`, Simmer uses state augmentation to ensure safety.
    Additionally, Simmer uses controller to control the safety budget.

    .. note::
        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    OmniSafe provides two implementations of Simmer RL: :class:`PPOSimmer` and :class:`TRPOSimmer`.

    References:
        - Title: Effects of Safety State Augmentation on Safe Exploration.
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
            David Mguni, Jun Wang, Haitham Bou-Ammar.
        - URL: `Simmer <https://arxiv.org/pdf/2206.02675.pdf>`_

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`SimmerAdapter`."""
        super(OnPolicyAdapter, self).__init__(env_id, num_envs, seed, cfgs)

        self._num_envs: int = num_envs
        self._safety_budget: torch.Tensor = (
            self._cfgs.algo_cfgs.safety_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
            * torch.ones(num_envs, 1)
        ).to(self._device)
        self._upper_budget: torch.Tensor = (
            self._cfgs.algo_cfgs.upper_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
            * torch.ones(num_envs, 1)
        ).to(self._device)
        self._rel_safety_budget: torch.Tensor = (self._safety_budget / self._upper_budget).to(
            self._device,
        )

        assert isinstance(self._env.observation_space, Box), 'Observation space must be Box'
        self._observation_space: Box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space.shape[0] + 1,),
        )
        self._controller: BaseSimmerAgent = SimmerPIDAgent(
            cfgs=cfgs.control_cfgs,
            budget_bound=self._upper_budget.cpu(),
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        .. note::
            Additionally, the safety observation will be reset. And the safety budget will be reset
            to the value of current ``rel_safety_budget``.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._safety_obs = self._rel_safety_budget * torch.ones(self._num_envs, 1).to(self._device)
        obs = self._augment_obs(obs)
        return obs, info

    def control_budget(self, ep_costs: torch.Tensor) -> None:
        """Control the safety budget.

        Args:
            ep_costs (torch.Tensor): The episode costs.
        """
        ep_costs = (
            ep_costs
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
        )
        self._safety_budget = self._controller.act(
            safety_budget=self._safety_budget.cpu(),
            observation=ep_costs.cpu(),
        ).to(self._device)
        self._rel_safety_budget = (self._safety_budget / self._upper_budget).to(self._device)
