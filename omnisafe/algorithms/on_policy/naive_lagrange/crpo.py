# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the on-policy CRPO algorithm."""

from typing import NamedTuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG


@registry.register
class OnCRPO(NaturalPG):
    """The on-policy CRPO algorithm.

    References:
        - Title: CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee.
        - Authors: Tengyu Xu, Yingbin Liang, Guanghui Lan.
        - URL: `CRPO <https://arxiv.org/pdf/2011.05869.pdf>`_.
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize CRPO.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        NaturalPG.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        self.rew_update = 0
        self.cost_update = 0

    def algorithm_specific_logs(self) -> None:
        """Log the CRPO specific information.

        .. list-table::

            *  -   Things to log
               -   Description
            *  -   Metrics/LagrangeMultiplier
               -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Misc/RewUpdate', self.rew_update)
        self.logger.log_tabular('Misc/CostUpdate', self.cost_update)

    def compute_surrogate(self, adv: torch.Tensor, cost_adv: torch.Tensor) -> torch.Tensor:
        """Compute the surrogate loss of the policy.

        In CRPO algorithm, we first judge whether the cost is within the limit.
        If the cost is within the limit, we use the advantage of the policy.
        Otherwise, we use the advantage of the cost.

        Args:
            adv (torch.Tensor): The advantage of the policy.
            cost_adv (torch.Tensor): The advantage of the cost.
        """
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        if Jc <= self.cfgs.cost_limit + self.cfgs.distance:
            self.rew_update += 1
            return adv
        self.cost_update += 1
        return -cost_adv
