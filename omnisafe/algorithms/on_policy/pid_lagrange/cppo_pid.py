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
"""Implementation of the CPPOPID (PID version of PPOLag) algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.pid_lagrange import PIDLagrangian


@registry.register
class CPPOPID(PPO):
    """The CPPOPID (PID version of PPOLag) algorithm.

    References:
        - Title: Responsive Safety in Reinforcement Learning by PID Lagrangian Methods
        - Authors: Adam Stooke, Joshua Achiam, Pieter Abbeel.
        - URL: `CPPOPID <https://arxiv.org/abs/2007.03964>`_
    """

    def _init(self) -> None:
        """Initialize the CPPOPID specific model.

        The CPPOPID algorithm uses a PID-Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: PIDLagrangian = PIDLagrangian(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the CPPOPID specific information.

        +----------------------------+------------------------------+
        | Things to log              | Description                  |
        +============================+==============================+
        | Metrics/LagrangeMultiplier | The PID-Lagrange multiplier. |
        +----------------------------+------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the PID-Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm.
            When a lagrange multiplier is used,
            the :meth:`_loss_pi` method will return the loss of the policy as:

            .. math::

                L_{\pi} = \mathbb{E}_{s_t \sim \rho_{\pi}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old} (a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the PID-Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes.
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update PID-Lagrange multiplier parameter
        self._lagrange.pid_update(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        CPPOPID uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The ``advantage`` combined with ``reward_advantage`` and ``cost_advantage``.
        """
        penalty = self._lagrange.lagrangian_multiplier
        return (adv_r - penalty * adv_c) / (1 + penalty)
