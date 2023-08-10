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
"""Offline Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch

from omnisafe.common.logger import Logger
from omnisafe.envs.core import make, support_envs
from omnisafe.envs.wrapper import ActionScale, TimeLimit
from omnisafe.models.base import Actor
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device


class OfflineAdapter:
    """Offline Adapter for OmniSafe.

    :class:`OfflineAdapter` is used to adapt the environment to the offline training.

    .. note::
        Technically, Offline training doesn't need env to interact with the agent.
        However, to visualize the performance of the agent when training,
        we still need instantiate a environment to evaluate the agent.
        OfflineAdapter provide an important interface ``evaluate`` to test the agent.

    Args:
        env_id (str): The environment id.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize a instance of :class:`OfflineAdapter`."""
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._env_id = env_id
        self._env = make(env_id, num_envs=1, device=cfgs.train_cfgs.device)
        self._cfgs = cfgs
        self._device = get_device(cfgs.train_cfgs.device)

        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, 1000, device=self._device)
        self._env = ActionScale(self._env, device=self._device, high=1.0, low=-1.0)

        self._env.set_seed(seed)

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._env.action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._env.observation_space

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

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
        return self._env.step(actions)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

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
        for _ in range(evaluate_epoisodes):
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0.0

            done = torch.Tensor([False])
            obs, _ = self.reset()
            while not done:
                action = agent.predict(obs.unsqueeze(0), deterministic=True)
                obs, reward, cost, terminated, truncated, _ = self.step(action.squeeze(0))

                ep_ret += reward.item()
                ep_cost += cost.item()
                ep_len += 1

                done = torch.logical_or(terminated, truncated)

            logger.store(
                {
                    'Metrics/EpRet': ep_ret,
                    'Metrics/EpCost': ep_cost,
                    'Metrics/EpLen': ep_len,
                },
            )
