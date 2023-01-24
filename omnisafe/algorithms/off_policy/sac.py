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

from typing import Dict, NamedTuple, Tuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG


@registry.register
# pylint: disable-next=too-many-instance-attributes
class SAC(DDPG):
    """The Soft Actor-Critic (SAC) algorithm.

    References:
        - Title: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
        - Authors: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine.
        - URL: `SAC <https://arxiv.org/abs/1801.01290>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize SAC."""
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )
        self.alpha = cfgs.alpha
        if self.cfgs.auto_alpha:
            self.target_entropy = -np.prod(self.env.action_space.shape)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.cfgs.alpha_lr)
            self.alpha = self.log_alpha.detach().exp()

    # pylint: disable-next=too-many-locals, too-many-arguments
    def compute_loss_v(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing value loss.

        .. note::

            The same as TD3, SAC uses two Q functions to reduce overestimation bias.
            In this function, we use the minimum of the two Q functions as the target Q value.

            Also, SAC use action with noise to compute the target Q value.

            Further more, SAC use the entropy of the action distribution to update Q value.

            .. math::
                L_1 = [Q^{C}_1 (s, a) - (r + \gamma (1-d) \underset{i = 1, 2}{\min} Q^{C}_i (s', \pi(s'))
                - \alpha \log p(a'|s')]^2\\
                L_2 = [Q^{C}_2 (s, a) - (r + \gamma (1-d) \underset{i = 1, 2}{\min} Q^{C}_i (s', \pi(s'))
                - \alpha \log p(a'|s')]^2

        Args:
            obs (torch.Tensor): ``observation`` saved in data.
            act (torch.Tensor): ``action`` saved in data.
            rew (torch.Tensor): ``reward`` saved in data.
            next_obs (torch.Tensor): ``next observation`` saved in data.
            done (torch.Tensor): ``terminated`` saved in data.
        """
        q_value_list = self.actor_critic.critic(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, _, logp_a_next = self.ac_targ.actor.predict(
                obs, deterministic=False, need_log_prob=True
            )
            q_targ = torch.min(torch.vstack(self.ac_targ.critic(next_obs, act_targ)), dim=0).values
            backup = rew + self.cfgs.gamma * (1 - done) * (q_targ - self.alpha * logp_a_next)
        # MSE loss against Bellman backup
        loss_q = []
        q_values = []
        for q_value in q_value_list:
            loss_q.append(torch.mean((q_value - backup) ** 2))
            q_values.append(torch.mean(q_value))
            self.logger.store(
                **{
                    'Train/RewardQValues': q_value.mean().item(),
                }
            )
        # useful info for logging
        q_info = dict(QVals=sum(q_values).detach().mean().item())
        return sum(loss_q), q_info

    def compute_loss_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computing ``pi/actor`` loss.

        .. note::

            SAC use the entropy of the action distribution to update policy.

        Args:
            obs (torch.Tensor): ``observation`` saved in data.
        """
        action, _, logp_a = self.actor_critic.actor.predict(
            obs, deterministic=False, need_log_prob=True
        )
        self.alpha_update(logp_a)

        loss_pi = (
            torch.min(
                self.actor_critic.critic(obs, action)[0], self.actor_critic.critic(obs, action)[1]
            )
            - self.alpha * logp_a
        )
        alpha_value = self.alpha.detach().mean().item() if self.cfgs.auto_alpha else self.alpha
        self.logger.store(
            **{
                'Misc/LogPi': logp_a.detach().mean().item(),
                'Misc/Alpha': alpha_value,
            }
        )
        pi_info = {}
        return -loss_pi.mean(), pi_info

    def alpha_update(self, log_prob) -> None:
        r"""Alpha discount.

        SAC discount alpha by ``alpha_gamma`` to decrease the entropy of the action distribution.
        At the end of each epoch, we have:

        .. math::
            \alpha \leftarrow \alpha \gamma_{\alpha}
        """
        if self.cfgs.auto_alpha:
            log_prob = log_prob.detach()
            log_prob += self.target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()
        else:
            self.alpha *= self.cfgs.alpha_gamma

    def algorithm_specific_logs(self) -> None:
        super().algorithm_specific_logs()
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/LogPi')
