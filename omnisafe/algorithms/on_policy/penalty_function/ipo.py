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
"""Implementation of IPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class IPO(PPO):
    """The Implementation of the IPO algorithm.

    References:
        - Title: IPO: Interior-point Policy Optimization under Constraints
        - Authors: Yongshuai Liu, Jiaxin Ding, Xin Liu.
        - URL: `IPO <https://arxiv.org/pdf/1910.09615.pdf>`_
    """

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/Penalty')

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        """Compute surrogate loss."""
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        penalty = self._cfgs.algo_cfgs.kappa / (self._cfgs.algo_cfgs.cost_limit - Jc + 1e-8)
        if penalty < 0 or penalty > self._cfgs.algo_cfgs.penalty_max:
            penalty = self._cfgs.algo_cfgs.penalty_max

        self._logger.store(**{'Misc/Penalty': penalty})

        return (adv_r - penalty * adv_c) / (1 + penalty)
