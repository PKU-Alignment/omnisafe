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

from typing import Dict, Optional

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self, env_id: str, num_envs: int, seed: int, cfgs: Config
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

        self._ep_ret: torch.Tensor
        self._ep_cost: torch.Tensor
        self._ep_len: torch.Tensor
        self._reset_log()

    def roll_out(  # pylint: disable=too-many-locals
        self,
        roll_out_step: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Roll out the environment and store the data in the buffer.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Agent.
            buf (VectorOnPolicyBuffer): Buffer.
            logger (Logger): Logger.
        """
        self._reset_log()

        obs, _ = self.reset()
        for step in range(roll_out_step):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.use_cost:
                logger.store(**{'Value/cost': value_c})
            logger.store(**{'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            dones = torch.logical_or(terminated, truncated)
            epoch_end = step >= roll_out_step - 1
            for idx, done in enumerate(dones):
                if epoch_end or done:
                    if epoch_end and not done:
                        logger.log(
                            f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.'
                        )
                        _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    elif done:
                        last_value_r = torch.zeros(1)
                        last_value_c = torch.zeros(1)

                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                    buffer.finish_path(last_value_r, last_value_c, idx)

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
