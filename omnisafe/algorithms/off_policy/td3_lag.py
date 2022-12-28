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
"""Implementation of the Lagrange version of the TD3 algorithm."""
from typing import NamedTuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.common.lagrange import Lagrange


@registry.register
class TD3Lag(TD3, Lagrange):  # pylint: disable=too-many-instance-attributes
    """The Lagrange version of the TD3 algorithm

    References:
        - Title: Addressing Function Approximation Error in Actor-Critic Methods
        - Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        - URL: `TD3 <https://arxiv.org/abs/1802.09477>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize TD3.

        Args:
            env_id (str): environment id.
            cfgs (dict): configurations.
            algo (str): algorithm name.
            wrapper_type (str): environment wrapper type.
        """
        TD3.__init__(
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

    def algorithm_specific_logs(self) -> None:
        """Log the TD3 Lag specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    def learn(self) -> torch.nn.Module:
        r"""
        This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.

        .. note::
            While a lagrange multiplier is used, the following steps are also performed:

            - :meth:`update_lagrange`: update the lagrange multiplier by:

            .. code-block:: python
                :linenos:

                    Jc = self.logger.get_stats('Metrics/EpCost')[0]
                    self.update_lagrange_multiplier(Jc)

            For details, please refer to the API documentation of ``Lagrange``.

        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        for steps in range(0, self.local_steps_per_epoch * self.epochs, self.update_every):
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            use_rand_action = steps < self.start_steps
            self.env.roll_out(
                self.actor_critic,
                self.buf,
                self.logger,
                deterministic=False,
                use_rand_action=use_rand_action,
                ep_steps=self.update_every,
            )

            # Update handling
            if steps >= self.update_after:
                for _ in range(self.update_every):
                    batch = self.buf.sample_batch()
                    self.update(data=batch)

            # End of epoch handling
            if steps % self.steps_per_epoch == 0 and steps:
                epoch = steps // self.steps_per_epoch
                if self.cfgs.use_cost and hasattr(self, 'lagrangian_multiplier'):
                    Jc = self.logger.get_stats('Metrics/EpCost')[0]
                    self.update_lagrange_multiplier(Jc)
                if self.cfgs.exploration_noise_anneal:
                    self.actor_critic.anneal_exploration(frac=epoch / self.epochs)
                # if self.cfgs.use_cost_critic:
                #     if self.use_cost_decay:
                #         self.cost_limit_decay(epoch)

                # Save model to disk
                if (epoch + 1) % self.cfgs.save_freq == 0:
                    self.logger.torch_save(itr=epoch)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()
                # Log info about epoch
                self.log(epoch, steps)
        return self.actor_critic

    def compute_loss_pi(self, obs: torch.Tensor) -> tuple((torch.Tensor, dict)):
        r"""Computing ``pi/actor`` loss.
        In the lagrange version of TD3, the loss is defined as:

        .. math::
            L = \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \pi(s)) - \lambda C(s, \pi(s))]

        where :math:`\lambda` is the lagrange multiplier.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
        """
        action = self.actor_critic.actor.predict(obs, deterministic=True, need_log_prob=False)
        loss_pi = self.actor_critic.critic(obs, action)[0]
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi -= self.lagrangian_multiplier * self.actor_critic.cost_critic(obs, action)[0]
        loss_pi /= 1 + penalty
        pi_info = {}
        return -loss_pi.mean(), pi_info
