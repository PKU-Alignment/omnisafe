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
"""Implementation of the CRPO algorithm."""

from typing import NamedTuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG


@registry.register
class CRPO(NaturalPG):
    """The CRPO algorithm."""

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

    def compute_surrogate(self, adv: torch.Tensor, cost_adv: torch.Tensor) -> torch.Tensor:
        """Compute the surrogate loss of the policy.

        In the CRPO algorithm, we first judge whether the cost is within the limit.
        If the cost is within the limit, we use the advantage of the policy.
        Otherwise, we use the advantage of the cost.

        Args:
            adv (torch.Tensor): The advantage of the policy.
            cost_adv (torch.Tensor): The advantage of the cost.
        """
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        if Jc <= self.cfgs.cost_limit + self.cfgs.distance:
            return adv
        return cost_adv
