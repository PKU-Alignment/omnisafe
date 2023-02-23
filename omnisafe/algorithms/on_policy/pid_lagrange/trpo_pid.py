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
"""Implementation of the PID-Lagrange version of the TRPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.pid_lagrange import PIDLagrangian


@registry.register
class TRPOPid(TRPO):
    """The PID-Lagrange version of the TRPO algorithm.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: https://arxiv.org/abs/2007.03964
    """

    def _init(self) -> None:
        super()._init()
        self._pid_lag = PIDLagrangian(**self._cfgs.PID_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('PID/pid_Kp')
        self._logger.register_key('PID/pid_Ki')
        self._logger.register_key('PID/pid_Kd')

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        penalty = self._pid_lag.cost_penalty
        return (adv_r - penalty * adv_c) / (1 + penalty)

    def _update(self) -> None:
        r"""Update actor, critic, running statistics as we used in the :class:`PolicyGradient` algorithm.

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
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._pid_lag.pid_update(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store(
            **{
                'Metrics/LagrangeMultiplier': self._pid_lag.cost_penalty,
                'PID/pid_Kp': self._pid_lag.pid_kp,
                'PID/pid_Ki': self._pid_lag.pid_ki,
                'PID/pid_Kd': self._pid_lag.pid_kd,
            }
        )
