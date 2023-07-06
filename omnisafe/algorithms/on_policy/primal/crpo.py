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
"""Implementation of the on-policy CRPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils.config import Config


@registry.register
class OnCRPO(TRPO):
    """The on-policy CRPO algorithm.

    References:
        - Title: CRPO: A New Approach for Safe Reinforcement Learning with Convergence Guarantee.
        - Authors: Tengyu Xu, Yingbin Liang, Guanghui Lan.
        - URL: `CRPO <https://arxiv.org/pdf/2011.05869.pdf>`_.
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of :class:`OnCRPO`."""
        super().__init__(env_id, cfgs)
        self._rew_update: int = 0
        self._cost_update: int = 0

    def _init_log(self) -> None:
        """Log the CRPO specific information.

        +-----------------+--------------------------------------------+
        | Things to log   | Description                                |
        +=================+============================================+
        | Misc/RewUpdate  | The number of times the reward is updated. |
        +-----------------+--------------------------------------------+
        | Misc/CostUpdate | The number of times the cost is updated.   |
        +-----------------+--------------------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Misc/RewUpdate')
        self._logger.register_key('Misc/CostUpdate')

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute the advantage surrogate.

        In CRPO algorithm, we first judge whether the cost is within the limit. If the cost is
        within the limit, we use the advantage of the policy. Otherwise, we use the advantage of the
        cost.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function chosen from reward and cost.
        """
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        if Jc <= self._cfgs.algo_cfgs.cost_limit + self._cfgs.algo_cfgs.distance:
            self._rew_update += 1
            return adv_r
        self._cost_update += 1
        self._logger.store(
            {
                'Misc/RewUpdate': self._rew_update,
                'Misc/CostUpdate': self._cost_update,
            },
        )
        return -adv_c
