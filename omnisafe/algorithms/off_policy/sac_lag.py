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
"""Implementation of the Lagrangian version of Soft Actor-Critic algorithm."""


import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SACLag(SAC):
    """The Lagrangian version of Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        action = self._actor_critic.actor.predict(obs, deterministic=False)
        log_prob = self._actor_critic.actor.log_prob(action)
        loss_q_r_1 = self._actor_critic.reward_critic(obs, action)[0].mean()
        loss_q_r_2 = self._actor_critic.reward_critic(obs, action)[1].mean()
        loss_r = (self._alpha * log_prob - torch.min(loss_q_r_1, loss_q_r_2)).mean()
        loss_q_c_1 = self._actor_critic.cost_critic(obs, action)[0].mean()
        loss_q_c_2 = self._actor_critic.cost_critic(obs, action)[1].mean()
        loss_c = self._lagrange.lagrangian_multiplier * torch.max(loss_q_c_1, loss_q_c_2)
        return loss_r + loss_c
    
    def _update_epoch(self) -> None:
        super()._update_epoch()
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._lagrange.update_lagrange_multiplier(Jc)
        self._logger.store(
            **{
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            }
        )

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            **{
                'Loss/Loss_cost_critic': 0.0,
                'Value/cost_critic': 0.0,
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.data.item(),
            }
        )
