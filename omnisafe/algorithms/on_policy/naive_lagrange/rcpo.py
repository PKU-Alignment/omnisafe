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
"""Implementation of the Reward Constrained Policy Optimization algorithm."""

from typing import NamedTuple

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG
from omnisafe.common.lagrange import Lagrange


@registry.register
class RCPO(NaturalPG, Lagrange):
    """Reward Constrained Policy Optimization.

    References:
        - Title: Reward Constrained Policy Optimization.
        - Authors: Chen Tessler, Daniel J. Mankowitz, Shie Mannor.
        - URL: `Reward Constrained Policy Optimization <https://arxiv.org/abs/1805.11074>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize RCPO.

        RCPO is a combination of :class:`NaturalPG` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        NaturalPG.__init__(
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

    def update(self) -> None:
        r"""Update actor, critic, running statistics as we used in the :class:`NaturalPG` algorithm.

        Additionally, we update the Lagrange multiplier parameter,
        by calling the :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`compute_loss_pi` is defined in the :class:`PolicyGradient` algorithm.
            When a lagrange multiplier is used,
            the :meth:`compute_loss_pi` method will return the loss of the policy as:

            .. math::
                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[ \frac{\pi_\theta(a_t|s_t)}{\pi_\theta^{old}(a_t|s_t)}
                [A^{R}(s_t, a_t) - \lambda A^{C}(s_t, a_t)] \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # pre-process data
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        raw_data, data = self.buf.pre_process_data()
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        obs, act, target_v, target_c, log_p, adv, cost_adv = (
            data['obs'],
            data['act'],
            data['target_v'],
            data['target_c'],
            data['log_p'],
            data['adv'],
            data['cost_adv'],
        )
        # Update critic
        self.update_value_net(obs=obs, target_v=target_v)
        self.update_cost_net(obs=obs, target_c=target_c)
        # Update actor
        self.update_policy_net(obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv)
        return raw_data, data

    def algorithm_specific_logs(self) -> None:
        """Log the RCPO specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
