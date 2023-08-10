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
"""Saute Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze
from omnisafe.utils.config import Config


class SauteAdapter(OnPolicyAdapter):
    """Saute Adapter for OmniSafe.

    Saute is a safe RL algorithm that uses state augmentation to ensure safety. The state
    augmentation is the concatenation of the original state and the safety state. The safety state
    is the safety budget minus the cost divided by the safety budget.

    .. note::
        - If the safety state is greater than 0, the reward is the original reward.
        - If the safety state is less than 0, the reward is the unsafe reward (always 0 or less than 0).

    OmniSafe provides two implementations of Saute RL: :class:`PPOSaute` and :class:`TRPOSaute`.

    References:
        - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
            David Mguni, Jun Wang, Haitham Bou-Ammar.
        - URL: `Saute <https://arxiv.org/abs/2202.06558>`_

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    _safety_obs: torch.Tensor
    _ep_budget: torch.Tensor

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`SauteAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)

        self._safety_budget: torch.Tensor = (
            self._cfgs.algo_cfgs.safety_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
            * torch.ones(num_envs, 1)
        ).to(self._device)

        assert isinstance(self._env.observation_space, Box), 'Observation space must be Box'
        self._observation_space: Box = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space.shape[0] + 1,),
        )

    @property
    def observation_space(self) -> Box:
        """The observation space of the environment."""
        return self._observation_space

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = False,
        cost_normalize: bool = False,
    ) -> None:
        """Wrapper the environment.

        .. warning::
            The reward or cost normalization is not supported in Saute Adapter.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to True.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, device=self._device, time_limit=1000)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if obs_normalize:
            self._env = ObsNormalize(self._env, device=self._device)
        assert reward_normalize is False, 'Reward normalization is not supported'
        assert cost_normalize is False, 'Cost normalization is not supported'
        self._env = ActionScale(self._env, device=self._device, low=-1.0, high=1.0)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        .. note::
            Additionally, the safety observation will be reset.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        self._safety_obs = torch.ones(self._env.num_envs, 1).to(self._device)
        obs = self._augment_obs(obs)
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Run one timestep of the environment's dynamics using the agent actions.

        .. note::
            The :meth:`_saute_step` will be called to update the safety observation. Then the reward
            will be updated by :meth:`_safety_reward`.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        next_obs, reward, cost, terminated, truncated, info = self._env.step(action)
        info['original_reward'] = reward

        self._safety_step(cost)
        reward = self._safety_reward(reward)

        # autoreset the environment
        done = torch.logical_or(terminated, truncated).float().unsqueeze(-1).float()
        self._safety_obs = self._safety_obs * (1 - done) + done

        augmented_obs = self._augment_obs(next_obs)

        if 'final_observation' in info:
            info['final_observation'] = self._augment_obs(info['final_observation'])

        return augmented_obs, reward, cost, terminated, truncated, info

    def _safety_step(self, cost: torch.Tensor) -> None:
        """Update the safety observation.

        Args:
            cost (torch.Tensor): The cost of the current step.
        """
        self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
        self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma

    def _safety_reward(self, reward: torch.Tensor) -> torch.Tensor:
        """Update the reward with the safety observation.

        .. note::
            If the safety observation is greater than 0, the reward will be the original reward.
            Otherwise, the reward will be the unsafe reward.

        Args:
            reward (torch.Tensor): The reward of the current step.

        Returns:
            The final reward determined by the safety observation.
        """
        safe = torch.as_tensor(self._safety_obs > 0, dtype=reward.dtype).squeeze(-1)
        return safe * reward + (1 - safe) * self._cfgs.algo_cfgs.unsafe_reward

    def _augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Augmenting the obs with the safety obs.

        The augmented obs is the concatenation of the original obs and the safety obs. The safety
        obs is the safety budget minus the cost divided by the safety budget.

        Args:
            obs (torch.Tensor): The original observation.

        Returns:
            The augmented observation.
        """
        return torch.cat([obs, self._safety_obs], dim=-1)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            Additionally, the safety observation will be updated and logged.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """
        super()._log_value(reward, cost, info)
        self._ep_budget += self._safety_obs.squeeze(-1)

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost, episode length and episode budget.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        super()._reset_log(idx)
        if idx is None:
            self._ep_budget = torch.zeros(self._env.num_envs).to(self._device)
        else:
            self._ep_budget[idx] = 0

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen`` and ``EpBudget``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        super()._log_metrics(logger, idx)
        logger.store({'Metrics/EpBudget': self._ep_budget[idx]})
