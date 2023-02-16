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

import torch

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models import ConstraintActorCritic
from omnisafe.utils.config import Config


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self, env_id: str, env_cls: str, num_envs: int, seed: int, cfgs: Config
    ) -> None:
        super().__init__(env_id, env_cls, num_envs, seed, cfgs)

        self._ep_ret = torch.zeros(
            self._env.num_envs,
        )
        self._ep_cost = torch.zeros(
            self._env.num_envs,
        )
        self._ep_len = torch.zeros(
            self._env.num_envs,
        )

    def roll_out(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buf: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Roll out the environment and store the data in the buffer.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Agent.
            buf (VectorOnPolicyBuffer): Buffer.
            logger (Logger): Logger.
        """
        obs, _ = self.reset()
        for step in range(steps_per_epoch):
            act, _, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._ep_ret += info.get('original_reward', reward)
            self._ep_cost += info.get('original_cost', cost)
            self._ep_len += 1

            if self._cfgs.use_cost:
                logger.store(**{'Value/cost': value_c})
            logger.store(**{'Value/reward': value_r})

            buf.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            dones = terminated or truncated
            epoch_end = step >= steps_per_epoch - 1
            for idx, done in enumerate(dones):
                if epoch_end and not done:
                    logger.log(
                        f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.'
                    )
                    _, _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                elif done:
                    last_value_r = torch.tensor(0.0)
                    last_value_c = torch.tensor(0.0)

                    logger.store(
                        **{
                            'Metrics/EpRet': self._ep_ret[idx],
                            'Metrics/EpCost': self._ep_cost[idx],
                            'Metrics/EpLen': self._ep_len[idx],
                        }
                    )

                    self._ep_ret[idx] = 0.0
                    self._ep_cost[idx] = 0.0
                    self._ep_len[idx] = 0.0

                buf.finish_path(last_value_r, last_value_c, idx)
