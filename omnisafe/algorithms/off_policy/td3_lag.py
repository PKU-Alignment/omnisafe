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

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.common.lagrange import Lagrange


@registry.register
class TD3Lag(TD3, Lagrange):  # pylint: disable=too-many-instance-attributes
    """The Lagrange version of the TD3 algorithm

    References:
        Title: Addressing Function Approximation Error in Actor-Critic Methods
        Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        URL: https://arxiv.org/abs/1802.09477
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
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

    def algorithm_specific_logs(self):
        """Use this method to collect log information."""
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Args:
            data (dict): data.

        Returns:
            torch.Tensor.
        """
        action = self.actor_critic.actor.predict(
            data['obs'], deterministic=True, need_log_prob=False
        )
        loss_pi = self.actor_critic.critic(data['obs'], action)[0]
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi -= (
            self.lagrangian_multiplier * self.actor_critic.cost_critic(data['obs'], action)[0]
        )
        loss_pi /= 1 + penalty
        pi_info = {}
        return -loss_pi.mean(), pi_info

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

        # Unfreeze Q-network so you can optimize it at next TD3 step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()
