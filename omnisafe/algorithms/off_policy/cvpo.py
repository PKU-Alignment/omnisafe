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
"""Implementation of the CVPO algorithm."""


import numpy as np
import torch
from scipy.optimize import minimize
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils.algo_utils import gaussian_kl
from omnisafe.utils.tools import to_ndarray


@registry.register
class CVPO(DDPG):  # pylint: disable=too-many-instance-attributes,too-many-locals
    """Constrained Variational Policy Optimization for Safe Reinforcement Learning.

    References:
        Title: Constrained Variational Policy Optimization for Safe Reinforcement Learning..
        Authors: Zuxin Liu, Zhepeng Cen, Vladislav Isenbaev,
                 Wei Liu, Zhiwei Steven Wu, Bo Li, Ding Zhao
        URL: https://arxiv.org/abs/2201.11927v2

    """

    def __init__(
        self,
        env_id: str,
        cfgs=None,
    ):
        """
        Constrained Variational Policy Optimization

        Args:
            env_id (str): Environment ID.
            cfgs (dict): Configuration dictionary.
        """

        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )

        self.kl_mean_constraint = self.cfgs.kl_mean_constraint
        self.kl_var_constraint = self.cfgs.kl_var_constraint
        self.kl_constraint = self.cfgs.kl_constraint
        self.alpha_mean_scale = self.cfgs.alpha_mean_scale
        self.alpha_var_scale = self.cfgs.alpha_var_scale
        self.alpha_scale = self.cfgs.alpha_scale
        self.alpha_mean_max = self.cfgs.alpha_mean_max
        self.alpha_var_max = self.cfgs.alpha_var_max
        self.alpha_max = self.cfgs.alpha_max
        self.sample_action_num = self.cfgs.sample_action_num
        self.mstep_iteration_num = self.cfgs.mstep_iteration_num
        self.dual_constraint = self.cfgs.dual_constraint
        self.eta = 0.1
        self.lam = 0.1
        self.alpha_mean = 0.0
        self.alpha_var = 0.0
        self.cost_limit = self.cfgs.cost_limit

    def update_policy_net(self, data) -> None:  # pylint: disable=too-many-locals
        """Update policy network.

        Args:
            data (dict): data dictionary.
        """
        obs = data['obs']  # [batch, obs_dim]
        num_action = self.sample_action_num
        num_obs = obs.shape[0]
        act_dim = self.actor_critic.act_dim
        obs_dim = self.actor_critic.obs_shape[0]

        with torch.no_grad():
            # sample N actions per state
            b_mean, b_var = self.ac_targ.actor.predict(
                obs, deterministic=True, need_log_prob=True
            )  # (K,)
            b_dist = MultivariateNormal(b_mean, scale_tril=b_var)
            sampled_actions = b_dist.sample((num_action,))

            expanded_states = obs[None, ...].expand(num_action, -1, -1)
            target_q = self.ac_targ.critic(
                expanded_states.reshape(-1, obs_dim), sampled_actions.reshape(-1, act_dim)
            )[0]
            target_q = target_q.reshape(num_action, num_obs)
            target_q_np = to_ndarray(target_q).T
            target_qc = self.ac_targ.cost_critic(
                expanded_states.reshape(-1, obs_dim), sampled_actions.reshape(-1, act_dim)
            )[0]
            target_qc = target_qc.reshape(num_action, num_obs)
            target_qc_np = to_ndarray(target_qc).T

        def dual(val):
            """dual function of the non-parametric variational."""
            beta, lam = val
            target_q_np_comb = target_q_np - lam * target_qc_np
            max_q = np.max(target_q_np_comb, 1)
            return (
                beta * self.dual_constraint
                + lam * self.cost_limit
                + np.mean(max_q)
                + beta
                * np.mean(
                    np.log(np.mean(np.exp((target_q_np_comb - max_q[:, None]) / beta), axis=1))
                )
            )

        bounds = [(1e-6, 1e5), (1e-6, 1e5)]
        options = {'ftol': 1e-3, 'maxiter': 10}
        res = minimize(
            dual,
            np.array([self.eta, self.lam]),
            method='SLSQP',
            bounds=bounds,
            tol=1e-3,
            options=options,
        )
        self.eta, self.lam = res.x

        raw_loss = torch.softmax((target_q - self.lam * target_qc) / self.eta, dim=0)

        # M-Step of Policy Improvement
        for _ in range(self.mstep_iteration_num):
            mean, var = self.actor_critic.actor.predict(obs, deterministic=True, need_log_prob=True)

            actor = MultivariateNormal(loc=mean, scale_tril=b_var)
            actor_ = MultivariateNormal(loc=b_mean, scale_tril=var)
            loss_p = torch.mean(
                raw_loss
                * (
                    actor.expand((num_action, num_obs)).log_prob(sampled_actions)
                    + actor_.expand((num_action, num_obs)).log_prob(sampled_actions)
                )
            )

            kl_mu, kl_sigma, _, sigma_det = gaussian_kl(
                mean_p=b_mean, mean_q=mean, var_p=b_var, var_q=var
            )

            if np.isnan(kl_mu.item()):
                raise RuntimeError('kl_mu is nan')
            if np.isnan(kl_sigma.item()):
                raise RuntimeError('kl_sigma is nan')

            # Update lagrange multipliers by gradient descent
            self.alpha_mean -= (
                self.alpha_mean_scale * (self.kl_mean_constraint - kl_mu).detach().item()
            )
            self.alpha_var -= (
                self.alpha_var_scale * (self.kl_var_constraint - kl_sigma).detach().item()
            )

            self.alpha_mean = np.clip(self.alpha_mean, 0.0, self.alpha_mean_max)
            self.alpha_var = np.clip(self.alpha_var, 0.0, self.alpha_var_max)

            self.actor_optimizer.zero_grad()
            loss_l = -(
                loss_p
                + self.alpha_mean * (self.kl_mean_constraint - kl_mu)
                + self.alpha_var * (self.kl_var_constraint - kl_sigma)
            )

            loss_l.backward()
            clip_grad_norm_(self.actor_critic.actor.parameters(), 0.01)
            self.actor_optimizer.step()
            self.logger.store(
                **{
                    'Loss/Pi': (-loss_p).item(),
                    'Misc/mean_sigma_det': sigma_det.item(),
                    'Misc/max_kl_sigma': kl_sigma.item(),
                    'Misc/max_kl_mu': kl_mu.item(),
                    'Misc/eta': self.eta,
                }
            )

    def algorithm_specific_logs(self):
        """Use this method to collect log information."""
        super().algorithm_specific_logs()
        self.logger.log_tabular('Misc/mean_sigma_det')
        self.logger.log_tabular('Misc/max_kl_sigma')
        self.logger.log_tabular('Misc/max_kl_mu')
        self.logger.log_tabular('Misc/eta')
