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
"""Implementation of the Lagrange version of the CRPO algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG


@registry.register
# pylint: disable-next=too-many-instance-attributes
class OffCRPO(DDPG):
    """The CRPO algorithm.

    References:
        - Title: CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee
        - Authors: Tengyu Xu, Yingbin Liang, Guanghui Lan.
        - URL: `CRPO <https://arxiv.org/pdf/2011.05869.pdf>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize CRPO."""
        DDPG.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        self.cost_limit = self.cfgs.init_cost_limit
        self.rew_update = 0
        self.cost_update = 0

    def _specific_init_logs(self):
        super()._specific_init_logs()
        self.logger.register_key('Loss/Loss_pi_c')
        self.logger.register_key('Misc/CostLimit')
        self.logger.register_key('Misc/RewUpdate')
        self.logger.register_key('Misc/CostUpdate')

    def algorithm_specific_logs(self) -> None:
        """Log the CRPO specific information.

        .. list-table::

            *  -   Things to log
               -   Description
            *  -   ``Loss/Loss_pi_c``
               -   The loss of the cost critic.
            *  -   ``Misc/CostLimit``
               -   The cost limit.
            *  -   ``Misc/RewUpdate``
               -   The number of reward updates.
            *  -   ``Misc/CostUpdate``
               -   The number of cost updates.
        """
        super().algorithm_specific_logs()
        self.logger.store(
            **{
                'Misc/CostLimit': self.cost_limit,
                'Misc/RewUpdate': self.rew_update,
                'Misc/CostUpdate': self.cost_update,
            }
        )

    def compute_loss_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing ``pi/actor`` loss.

        In CRPO algorithm, the loss function is defined as:

        .. math::

            \mathcal{L}_{\pi} = - Q^V(s, \pi(s)) \text{ if } \mathcal{L}_{\pi_c} \leq \mathcal{L}_{\text{limit}} \\

            \mathcal{L}_{\pi} = Q^C(s, \pi(s)) \text{ if } \mathcal{L}_{\pi_c} > \mathcal{L}_{\text{limit}}

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
        """
        _, action = self.actor_critic.actor.predict(obs, deterministic=False, need_log_prob=False)
        loss_pi_c = self.actor_critic.cost_critic(obs, action)[0]
        loss_pi_c = F.relu(loss_pi_c - self.cost_limit)
        if loss_pi_c.mean().item() > self.cost_limit:
            loss_pi = -loss_pi_c
            self.cost_update += 1
        else:
            loss_pi = self.actor_critic.critic(obs, action)[0]
            self.rew_update += 1
        pi_info = {}
        self.logger.store(
            **{
                'Loss/Loss_pi_c': loss_pi_c.mean().item(),
            }
        )
        return -loss_pi.mean(), pi_info
