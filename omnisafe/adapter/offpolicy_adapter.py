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

from __future__ import annotations

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config


class OffPolicyAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe.

    :class:`OffPolicyAdapter` is used to adapt the environment to the off-policy training.

    .. note::

        Off-policy training need to update the policy before finish the episode,
        so the :class:`OffPolicyAdapter` will store the current observation in ``_current_obs``.
        After update the policy, the agent will *remember* the current observation and
        use it to interact with the environment.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.

    Attributes:
        _env_id (str): The environment id.
        _env (CMDP): The environment.
        _cfgs (Config): The configuration.
        _device (torch.device): The device.
        _ep_ret (torch.Tensor): The episode return.
        _ep_cost (torch.Tensor): The episode cost.
        _ep_len (torch.Tensor): The episode length.
        _current_obs (torch.Tensor): The current observation.
        _max_ep_len (int): The maximum episode length.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        super().__init__(env_id, num_envs, seed, cfgs)

        self._ep_ret: torch.Tensor
        self._ep_cost: torch.Tensor
        self._ep_len: torch.Tensor
        self._current_obs, _ = self.reset()
        self._max_ep_len = 1000
        self._device = cfgs.train_cfgs.device
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

        .. warning::

            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Agent.
            buf (VectorOnPolicyBuffer): Buffer.
            logger (Logger): Logger.
        """
        for _ in range(roll_out_step):
            if use_rand_action:
                act = torch.as_tensor(self._env.sample_action(), dtype=torch.float32).to(
                    self._device,
                )
            else:
                act = agent.step(self._current_obs, deterministic=False)

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
        info: dict,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The reward.
            cost (torch.Tensor): The cost.
            **kwargs: Other arguments.
        """
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += 1

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics.

        Args:
            logger (Logger): Logger.
            idx (int): The index of the environment.
        """
        logger.store(
            **{
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
            },
        )

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset log.

        Args:
            idx (int | None): The index of the environment.
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0
