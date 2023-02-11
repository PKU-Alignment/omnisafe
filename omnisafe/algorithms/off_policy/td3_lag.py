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
"""Implementation of the Lagrange version of the TD3 algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.common.lagrange import Lagrange


@registry.register
# pylint: disable-next=too-many-instance-attributes
class TD3Lag(TD3, Lagrange):
    """The Lagrange version of the TD3 algorithm

    References:
        - Title: Addressing Function Approximation Error in Actor-Critic Methods
        - Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        - URL: `TD3 <https://arxiv.org/abs/1802.09477>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize Lagrange version of the TD3.

        Args:
            env_id (str): environment id.
            cfgs (dict): configurations.
            algo (str): algorithm name.
            wrapper_type (str): environment wrapper type.
        """
        TD3.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        Lagrange.__init__(self, **self.cfgs.lagrange_cfgs)

    def _specific_init_logs(self):
        super()._specific_init_logs()
        self.logger.register_key('Metrics/LagrangeMultiplier')
        self.logger.register_key('Loss/Loss_pi_c')
        self.logger.register_key('Misc/CostLimit')

    def algorithm_specific_logs(self) -> None:
        """Log the TD3 Lag specific information.

        .. list-table::

            *  -   Things to log
               -   Description
            *  -   Metrics/LagrangeMultiplier
               -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.store(
            **{
                'Metrics/LagrangeMultiplier': self.lagrangian_multiplier.item(),
                'Misc/CostLimit': self.cost_limit,
            }
        )

    def compute_loss_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing ``pi/actor`` loss.

        In the lagrange version of TD3, the loss is defined as:

        .. math::
            L = \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \pi(s)) - \lambda C(s, \pi(s))]

        where :math:`\lambda` is the lagrange multiplier.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
        """
        action, _ = self.actor_critic.actor.predict(obs, deterministic=False, need_log_prob=False)
        loss_pi = torch.min(
            self.actor_critic.critic(obs, action)[0], self.actor_critic.critic(obs, action)[1]
        )
        loss_pi_c = self.actor_critic.cost_critic(obs, action)[0]
        loss_pi_c = F.relu(loss_pi_c - self.cost_limit)
        self.update_lagrange_multiplier(loss_pi_c.mean().item())
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi -= penalty * loss_pi_c
        loss_pi /= 1 + penalty
        pi_info = {}
        self.logger.store(
            **{
                'Loss/Loss_pi_c': loss_pi_c.mean().item(),
            }
        )
        return -loss_pi.mean(), pi_info
