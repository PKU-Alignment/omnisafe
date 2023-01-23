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

from typing import Dict, NamedTuple, Tuple

import torch
import torch.nn.functional as F

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.common.pid_lagrange import PIDLagrangian
from omnisafe.utils.config_utils import namedtuple2dict


@registry.register
# pylint: disable-next=too-many-instance-attributes
class SACPid(SAC, PIDLagrangian):  # pylint: disable-next=too-many-instance-attributes
    """The Pid-Lagrangian version of SAC algorithm.

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
        # self.cost_limit=self.set_cost_limit(self.cfgs.lagrange_cfgs.cost_limit)
        PIDLagrangian.__init__(self, **namedtuple2dict(self.cfgs.PID_cfgs))

    def cost_limit_decay(
        self,
        epoch: int,
        end_epoch: int,
    ) -> None:
        """Decay cost limit."""
        if epoch < end_epoch:
            self.cost_limit = (
                self.cfgs.init_cost_limit * (1 - epoch / end_epoch)
                + self.cfgs.target_cost_limit * epoch / end_epoch
            )
            self.cost_limit /= (1 - self.cfgs.gamma**self.max_ep_len) / (1 - self.cfgs.gamma)

    def algorithm_specific_logs(self) -> None:
        """Log the SAC Lag specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.cost_penalty)
        self.logger.log_tabular('Loss/Loss_pi_c')
        self.logger.log_tabular('Misc/CostLimit', self.cost_limit)
        self.logger.log_tabular('PID/pid_Kp', self.pid_kp)
        self.logger.log_tabular('PID/pid_Ki', self.pid_ki)
        self.logger.log_tabular('PID/pid_Kd', self.pid_kd)

    def compute_loss_pi(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing ``pi/actor`` loss.

        In the lagrange version of DDPG, the loss is defined as:

        .. math::
            L = \mathbb{E}_{s \sim \mathcal{D}} [ Q(s, \pi(s)) - \lambda C(s, \pi(s)) - \mu \log \pi(s)]

        where :math:`\lambda` is the lagrange multiplier, :math:`\mu` is the entropy coefficient.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
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
        loss_pi_c = self.actor_critic.cost_critic(obs, action)[0]
        loss_pi_c = F.relu(loss_pi_c - self.cost_limit)
        self.pid_update(loss_pi_c.mean().item())
        loss_pi -= self.cost_penalty * loss_pi_c
        loss_pi /= 1 + self.cost_penalty
        pi_info = {}
        self.logger.store(
            **{
                'Loss/Loss_pi_c': loss_pi_c.mean().item(),
                'Misc/LogPi': logp_a.detach().mean().item(),
                'Misc/Alpha': self.alpha.detach().mean().item(),
            }
        )
        pi_info = {}
        return -loss_pi.mean(), pi_info

    # pylint: disable-next=too-many-arguments
    def compute_loss_c(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        cost: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
            next_obs (torch.Tensor): ``next observation`` saved in data.
            done (torch.Tensor): ``terminated`` saved in data.
        """
        cost_q_value = self.actor_critic.cost_critic(obs, act)[0]
        self.logger.store(
            **{
                'Train/CostQValues': cost_q_value.mean().item(),
            }
        )
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ, _, logp_a_next = self.ac_targ.actor.predict(
                next_obs, deterministic=False, need_log_prob=True
            )
            qc_targ = self.ac_targ.cost_critic(next_obs, act_targ)[0]
            backup = cost + self.cfgs.gamma * (qc_targ - self.alpha * logp_a_next)
        # MSE loss against Bellman backup
        loss_qc = ((cost_q_value - backup) ** 2).mean()
        # useful info for logging
        qc_info = dict(QCosts=cost_q_value.detach().numpy())

        return loss_qc, qc_info
