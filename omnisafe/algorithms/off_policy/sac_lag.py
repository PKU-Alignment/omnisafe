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
"""Implementation of the Lagrange version of the SAC algorithm."""
from typing import NamedTuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange


@registry.register
class SACLag(SAC, Lagrange):  # pylint: disable=too-many-instance-attributes
    """The Lagrange version of SAC algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize SACLag.

        Args:
            env_id (str): environment id.
            cfgs (dict): configuration.
            algo (str): algorithm name.
            wrapper_type (str): environment wrapper type.
        """
        SAC.__init__(
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
        """Log the SAC Lag specific information.

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
        In the lagrange version of DDPG, the loss is defined as:

        .. math::
            L = \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \pi(s)) - \lambda C(s, \pi(s)) - \mu \log \pi(s)]

        where :math:`\lambda` is the lagrange multiplier, :math:`\mu` is the entropy coefficient.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
        """
        action, logp_a = self.actor_critic.actor.predict(
            obs, deterministic=True, need_log_prob=True
        )
        loss_pi = self.actor_critic.critic(obs, action)[0] - self.alpha * logp_a
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi -= self.lagrangian_multiplier * self.actor_critic.cost_critic(obs, action)[0]
        loss_pi /= 1 + penalty
        pi_info = {}
        return -loss_pi.mean(), pi_info

    # pylint: disable=too-many-arguments
    def compute_loss_c(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        obs_next: torch.Tensor,
        done: torch.Tensor,
    ) -> tuple((torch.Tensor, dict)):
        r"""Computing cost loss.

        .. note::

            The same as TD3, SAC uses two Q functions to reduce overestimation bias.
            In this function, we use the minimum of the two Q functions as the target Q value.

            Also, SAC use action with noise to compute the target Q value.

            Further more, SAC use the entropy of the action distribution to update Q value.

        Args:
            obs (torch.Tensor): ``observation`` saved in data.
            act (torch.Tensor): ``action`` saved in data.
            rew (torch.Tensor): ``reward`` saved in data.
            obs_next (torch.Tensor): ``next observations`` saved in data.
            done (torch.Tensor): ``terminated`` saved in data.
        """
        cost_q_value = self.actor_critic.cost_critic(obs, act)[0]

        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, logp_a_next = self.ac_targ.actor.predict(
                obs_next, deterministic=False, need_log_prob=True
            )
            qc_targ = self.ac_targ.cost_critic(obs_next, act_targ)[0]
            backup = cost + self.cfgs.gamma * (1 - done) * (qc_targ - self.alpha * logp_a_next)
        # MSE loss against Bellman backup
        loss_qc = ((cost_q_value - backup) ** 2).mean()
        # Useful info for logging
        qc_info = dict(QCosts=cost_q_value.detach().numpy())

        return loss_qc, qc_info
