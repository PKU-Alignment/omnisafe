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
"""Offline Adapter for OmniSafe."""

from typing import Dict, Tuple

import torch

from omnisafe.common.logger import Logger
from omnisafe.envs.core import make, support_envs
from omnisafe.envs.wrapper import ActionScale, TimeLimit
from omnisafe.models.base import Actor
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config


class OfflineAdapter:
    """Offline Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        seed: int,
        cfgs: Config,
    ) -> None:
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._env_id = env_id
        self._env = make(env_id, num_envs=1)
        self._cfgs = cfgs
        self._device = cfgs.train_cfgs.device

        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, 1000, device=self._device)
        self._env = ActionScale(self._env, device=self._device, high=1.0, low=-1.0)

        self._env.set_seed(seed)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment.

        Returns:
            OmnisafeSpace: the action space.
        """
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment.

        Returns:
            OmnisafeSpace: the observation space.
        """
        return self._env.observation_space

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Take a step in the environment."""
        return self._env.step(actions)

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        """Reset the environment."""
        return self._env.reset()

    def evaluate(
        self,
        evaluate_epoisodes: int,
        agent: Actor,
        logger: Logger,
    ) -> None:
        """Evaluate the agent in the environment.

        Args:
            evaluate_epoisodes (int): the number of episodes for evaluation.
            agent (Actor): the agent to be evaluated.
            logger (Logger): the logger for logging the evaluation results.
        """

        agent = agent.to('cpu')

        for _ in range(evaluate_epoisodes):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0.0

            done = torch.Tensor([False])
            obs, _ = self.reset()
            while not done:
                action = agent.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, cost, terminated, truncated, _ = self.step(action)

                ep_ret += reward.item()
                ep_cost += cost.item()
                ep_len += 1

                done = torch.logical_or(terminated, truncated)

            logger.store(
                **{
                    'Metrics/EpRet': ep_ret,
                    'Metrics/EpCost': ep_cost,
                    'Metrics/EpLen': ep_len,
                }
            )

        agent = agent.to(self._device)
