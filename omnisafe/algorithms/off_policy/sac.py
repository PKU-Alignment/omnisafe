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
"""Implementation of the SAC algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG


@registry.register
class SAC(DDPG):  # pylint: disable=too-many-instance-attributes
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        URL: https://arxiv.org/abs/1801.01290
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
        """Initialize SAC."""
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )
        self.alpha = cfgs.alpha
        self.alpha_gamma = cfgs.alpha_gamma

    # pylint: disable=too-many-locals
    def compute_loss_v(self, data):
        """Computing value loss.

        Args:
            data (dict): data from replay buffer.

        Returns:
            torch.Tensor.
        """
        obs, act, rew, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
        q_value_list = self.actor_critic.critic(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, logp_a_next = self.ac_targ.actor.predict(
                obs, deterministic=False, need_log_prob=True
            )
            q_targ = torch.min(torch.vstack(self.ac_targ.critic(obs_next, act_targ)), dim=0).values
            backup = rew + self.cfgs.gamma * (1 - done) * (q_targ - self.alpha * logp_a_next)
        # MSE loss against Bellman backup
        loss_q = []
        q_values = []
        for q_value in q_value_list:
            loss_q.append(torch.mean((q_value - backup) ** 2))
            q_values.append(torch.mean(q_value))

        # Useful info for logging
        q_info = dict(QVals=sum(q_values).detach().numpy())
        return sum(loss_q), q_info

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Args:
            data (dict): data from replay buffer.

        Returns:
            torch.Tensor.
        """
        action, logp_a = self.actor_critic.actor.predict(
            data['obs'], deterministic=True, need_log_prob=True
        )
        loss_pi = self.actor_critic.critic(data['obs'], action)[0] - self.alpha * logp_a
        pi_info = {'LogPi': logp_a.detach().numpy()}
        return -loss_pi.mean(), pi_info

    def update(self, data):
        """Update."""
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

        # Next run one gradient descent step for actor.
        self.update_policy_net(data)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()
        self.alpha_discount()

    def alpha_discount(self):
        """Alpha discount."""
        self.alpha *= self.alpha_gamma
