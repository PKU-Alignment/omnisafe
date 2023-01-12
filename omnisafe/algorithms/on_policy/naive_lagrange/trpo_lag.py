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
"""Implementation of the Lagrange version of the TRPO algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.lagrange import Lagrange


@registry.register
class TRPOLag(TRPO, Lagrange):
    """The Lagrange version of the TRPO algorithm.

    A simple combination of the Lagrange method and the Trust Region Policy Optimization algorithm.
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize TRPOLag.

        TRPOLag is a combination of :class:`TRPO` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        TRPO.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        Lagrange.__init__(
            self,
            cost_limit=self.cfgs.lagrange_cfgs.cost_limit,
            lagrangian_multiplier_init=self.cfgs.lagrange_cfgs.lagrangian_multiplier_init,
            lambda_lr=self.cfgs.lagrange_cfgs.lambda_lr,
            lambda_optimizer=self.cfgs.lagrange_cfgs.lambda_optimizer,
        )

    def update(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        r"""Update actor, critic, running statistics as we used in the :class:`TRPO` algorithm.

        Additionally, we update the Lagrange multiplier parameter,
        by calling the :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`compute_loss_pi` method is defined in the :class:`PolicyGradient` algorithm.
            When a lagrange multiplier is used,
            the :meth:`compute_loss_pi` method will return the loss of the policy as:

            .. math::
                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_\theta^{old}(a_t|s_t)}
                [A^{R}(s_t, a_t) - \lambda A^{C}(s_t, a_t)] \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        TRPO.update(self)

    def compute_surrogate(
        self,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surrogate loss.

        Policy Gradient only use reward advantage.

        Args:
            adv (torch.Tensor): reward advantage
            cost_adv (torch.Tensor): cost advantage
        """
        return adv - self.lagrangian_multiplier * cost_adv

    def algorithm_specific_logs(self) -> None:
        """Log the TRPOLag specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
