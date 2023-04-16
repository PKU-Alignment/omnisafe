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

from __future__ import annotations

import numpy as np
import torch
from gymnasium.spaces import Box

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import ActionScale, AutoReset, ObsNormalize, TimeLimit, Unsqueeze
from omnisafe.utils.config import Config


class SauteAdapter(OnPolicyAdapter):
    """OnPolicy Adapter for OmniSafe."""

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

        self._safety_budget: torch.Tensor
        self._safety_obs: torch.Tensor

        self._safety_budget = (
            self._cfgs.algo_cfgs.safety_budget
            * (1 - self._cfgs.algo_cfgs.saute_gamma**self._cfgs.algo_cfgs.max_ep_len)
            / (1 - self._cfgs.algo_cfgs.saute_gamma)
            / self._cfgs.algo_cfgs.max_ep_len
            * torch.ones(num_envs, 1)
        )

        self._ep_budget: torch.Tensor

        assert isinstance(self._env.observation_space, Box), 'Observation space must be Box'
        self._observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._env.observation_space.shape[0] + 1,),
        )

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = False,
        cost_normalize: bool = False,
    ):
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

    def reset(self) -> tuple[torch.Tensor, dict]:
        obs, info = self._env.reset()
        self._safety_obs = torch.ones(self._env.num_envs, 1)
        obs = self._augment_obs(obs)
        return obs, info

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
        self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
        self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma

    def _safety_reward(self, reward: torch.Tensor) -> torch.Tensor:
        safe = torch.as_tensor(self._safety_obs > 0, dtype=reward.dtype).squeeze(-1)
        return safe * reward + (1 - safe) * self._cfgs.algo_cfgs.unsafe_reward

    def _augment_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, self._safety_obs], dim=-1)

    def _log_value(self, reward: torch.Tensor, cost: torch.Tensor, info: dict, **kwargs) -> None:
        super()._log_value(reward, cost, info, **kwargs)
        self._ep_budget += self._safety_obs.squeeze(-1)

    def _reset_log(self, idx: int | None = None) -> None:
        super()._reset_log(idx)
        if idx is None:
            self._ep_budget = torch.zeros(self._env.num_envs)
        else:
            self._ep_budget[idx] = 0

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        super()._log_metrics(logger, idx)
        logger.store(**{'Metrics/EpBudget': self._ep_budget[idx]})
