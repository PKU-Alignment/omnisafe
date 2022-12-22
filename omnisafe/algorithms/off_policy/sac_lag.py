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
"""Implementation of the SACLag algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.lagrange import Lagrange


@registry.register
class SACLag(SAC, Lagrange):  # pylint: disable=too-many-instance-attributes
    """The Lagrange version of SAC algorithm.

    References:
        Paper Name: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        Paper author: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine
        Paper URL: https://arxiv.org/abs/1801.01290

    """

    def __init__(
        self,
        env_id: str,
        cfgs=None,
        algo: str = 'SAC-Lag',
        wrapper_type: str = 'OffPolicyEnvWrapper',
    ):
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
            algo=algo,
            wrapper_type=wrapper_type,
        )

        Lagrange.__init__(
            self,
            cost_limit=self.cfgs.lagrange_cfgs.cost_limit,
            lagrangian_multiplier_init=self.cfgs.lagrange_cfgs.lagrangian_multiplier_init,
            lambda_lr=self.cfgs.lagrange_cfgs.lambda_lr,
            lambda_optimizer=self.cfgs.lagrange_cfgs.lambda_optimizer,
        )

    def algorithm_specific_logs(self):
        """Use this method to collect log information."""
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Returns:
            torch.Tensor.
        """
        action, logp_a = self.actor_critic.actor.predict(
            data['obs'], deterministic=True, need_log_prob=True
        )
        loss_pi = self.actor_critic.critic(data['obs'], action)[0] - self.alpha * logp_a
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi -= (
            self.lagrangian_multiplier * self.actor_critic.cost_critic(data['obs'], action)[0]
        )
        loss_pi /= 1 + penalty
        pi_info = {}
        return -loss_pi.mean(), pi_info

    def compute_loss_c(self, data):
        """Computing cost loss.

        Returns:
            torch.Tensor.
        """
        obs, act, cost, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
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

    def update(self, data):
        """Update."""
        Jc = data['cost'].sum().item()
        self.update_lagrange_multiplier(Jc)
        # First run one gradient descent step for Q.
        self.update_value_net(data)
        if self.cfgs.use_cost:
            self.update_cost_net(data)
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = False

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for pi.
        self.update_policy_net(data)

        # Unfreeze Q-network so you can optimize it at next SAC step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()
        self.alpha_discount()
