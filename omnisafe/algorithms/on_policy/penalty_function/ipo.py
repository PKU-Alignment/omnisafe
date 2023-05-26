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
        """Log the IPO specific information.

        +---------------+--------------------------+
        | Things to log | Description              |
        +===============+==========================+
        | Misc/Penalty  | The penalty coefficient. |
        +---------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Misc/Penalty')

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        IPO uses the following surrogate loss:

        .. math::

            L = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                \frac{\pi_{\theta}^{'} (a_t|s_t)}{\pi_{\theta} (a_t|s_t)} A (s_t, a_t)
                - \kappa \frac{J^{C}_{\pi_{\theta}} (s_t, a_t)}{C - J^{C}_{\pi_{\theta}} (s_t, a_t) + \epsilon}
            \right]

        Where :math:`\kappa` is the penalty coefficient, :math:`C` is the cost limit,
        and :math:`\epsilon` is a small number to avoid division by zero.

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        penalty = self._cfgs.algo_cfgs.kappa / (self._cfgs.algo_cfgs.cost_limit - Jc + 1e-8)
        if penalty < 0 or penalty > self._cfgs.algo_cfgs.penalty_max:
            penalty = self._cfgs.algo_cfgs.penalty_max

        self._logger.store({'Misc/Penalty': penalty})

        return (adv_r - penalty * adv_c) / (1 + penalty)
