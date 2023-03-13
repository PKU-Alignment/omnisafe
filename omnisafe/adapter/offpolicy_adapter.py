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
"""OffPolicy Adapter for OmniSafe."""

from functools import partial
from typing import Dict, Optional

import torch
from gymnasium import spaces

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self, env_id: str, num_envs: int, seed: int, cfgs: Config
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

        self._ep_ret: torch.Tensor
        self._ep_cost: torch.Tensor
        self._ep_len: torch.Tensor
        self._current_obs, _ = self.reset()
        self._max_ep_len = 1000
        self._reset_log()

    def roll_out(  # pylint: disable=too-many-locals
        self,
        roll_out_step: int,
        agent: ConstraintActorQCritic,
        buffer: VectorOffPolicyBuffer,
        logger: Logger,
        use_rand_action: bool,
    ) -> None:
        """Roll out the environment and store the data in the buffer.

        Args:
            roll_out_step (int): Number of steps to roll out.
            agent (ConstraintActorCritic): Agent.
            buf (VectorOnPolicyBuffer): Buffer.
            logger (Logger): Logger.
            use_rand_action (bool): Whether to use random action.
        """
        if use_rand_action:
            if isinstance(self._env.action_space, spaces.Box):
                act_fn = partial(
                    torch.rand, size=(self._env.num_envs, *self._env.action_space.shape)
                )
        else:
            act_fn = partial(agent.step, self._current_obs, deterministic=False)

        for _ in range(roll_out_step):
            act = act_fn()
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            buffer.store(
                obs=self._current_obs,
                act=act,
                reward=reward,
                cost=cost,
                done=terminated,
                next_obs=next_obs,
            )

            self._current_obs = next_obs
            for idx, done in enumerate(torch.logical_or(terminated, truncated)):
                if done or self._ep_len[idx] >= self._max_ep_len:
                    # self.reset()
                    self._log_metrics(logger, idx)
                    self._reset_log(idx)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: Dict,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Log value."""
        self._ep_ret += info.get('original_reward', reward)
        self._ep_cost += info.get('original_cost', cost)
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics."""
        logger.store(
            **{
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            }
        )

    def _reset_log(self, idx: Optional[int] = None) -> None:
        """Reset log."""
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
