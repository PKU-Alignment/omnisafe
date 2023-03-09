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
import torch.nn as nn
from scipy.optimize import minimize
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from omnisafe.utils.math import gaussian_kl
from omnisafe.utils.tools import to_ndarray


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-many-locals
class CVPO(DDPG):
    """Constrained Variational Policy Optimization for Safe Reinforcement Learning.

    References:

        - Title: Constrained Variational Policy Optimization for Safe Reinforcement Learning.
        - Authors: Zuxin Liu, Zhepeng Cen, Vladislav Isenbaev,
                Wei Liu, Zhiwei Steven Wu, Bo Li, Ding Zhao
        - URL: https://arxiv.org/abs/2201.11927v2
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        super().__init__(env_id, cfgs)
        self._eta: float
        self._lam: float
        self._alpha_mean: float
        self._alpha_var: float

    def _init(self) -> None:
        super()._init()
        self._eta = 0.1
        self._lam = 0.1
        self._alpha_mean = 0.0
        self._alpha_var = 0.0

    # pylint: disable-next=too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
    ) -> None:
        num_action = self._cfgs.sample_action_num
        num_obs = obs.shape[0]
        act_dim = self.actor_critic.act_dim
        obs_dim = self.actor_critic.obs_shape[0]

        with torch.no_grad():
            # sample N actions per state
            b_dist = self._actor_critic.target_actor(obs)
            b_mean = b_dist.loc
            b_var = b_dist.scale_tril
            sampled_actions = b_dist.sample((num_action,))

            expanded_states = obs[None, ...].expand(num_action, -1, -1)
            target_q_r_1, target_q_r_2 = self._actor_critic.target_reward_critic(
                expanded_states.reshape(-1, obs_dim), sampled_actions.reshape(-1, act_dim)
            )
            target_q_r = torch.min(target_q_r_1, target_q_r_2)
            target_q_r = target_q_r.reshape(num_action, num_obs)
            target_q_r_np = to_ndarray(target_q_r).T

            target_q_c_1, target_q_c_2 = self._actor_critic.target_cost_critic(
                expanded_states.reshape(-1, obs_dim), sampled_actions.reshape(-1, act_dim)
            )
            target_q_c = torch.min(target_q_c_1, target_q_c_2)
            target_q_c = target_q_c.reshape(num_action, num_obs)
            target_q_c_np = to_ndarray(target_q_c).T

        def dual(val):
            """Dual function of the non-parametric variational."""
            beta, lam = val
            target_q_np_comb = target_q_r_np - lam * target_q_c_np
            max_q = np.max(target_q_np_comb, 1)
            return (
                beta * self.cfgs.dual_constraint
                + lam * self.cfgs.algo_cfgs.cost_limit
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
            np.array([self._eta, self._lam]),
            method='SLSQP',
            bounds=bounds,
            tol=1e-3,
            options=options,
        )
        self._eta, self._lam = res.x

        raw_loss = torch.softmax((target_q_r - self._lam * target_q_c) / self._eta, dim=0)

        # M-Step of Policy Improvement
        for _ in range(self.cfgs.mstep_iteration_num):
            mean, _, var = self.actor_critic.actor.predict(
                obs, deterministic=True, need_log_prob=True
            )

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

            # update lagrange multipliers by gradient descent
            self._alpha_mean -= (
                self.cfgs.alpha_mean_scale * (self.cfgs.kl_mean_constraint - kl_mu).detach().item()
            )
            self._alpha_var -= (
                self.cfgs.alpha_var_scale * (self.cfgs.kl_var_constraint - kl_sigma).detach().item()
            )

            self._alpha_mean = np.clip(self._alpha_mean, 0.0, self.cfgs.alpha_mean_max)
            self._alpha_var = np.clip(self._alpha_var, 0.0, self.cfgs.alpha_var_max)
            self.actor_optimizer.zero_grad()
            loss_l = -(
                loss_p
                + self._alpha_mean * (self.cfgs.kl_mean_constraint - kl_mu)
                + self._alpha_var * (self.cfgs.kl_var_constraint - kl_sigma)
            )
            loss_l.backward()
            clip_grad_norm_(self.actor_critic.actor.parameters(), 0.01)
            self.actor_optimizer.step()
            self.logger.store(
                **{
                    'Loss/Loss_pi': loss_p.mean().item(),
                    'Loss/Loss_l': loss_l.mean().item(),
                    'Train/mean_sigma_det': sigma_det.item(),
                    'Train/max_kl_sigma': kl_sigma.item(),
                    'Train/max_kl_mu': kl_mu.item(),
                    'Train/eta': self._eta,
                }
            )

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Loss/Loss_l')
        self._logger.register_key('Train/mean_sigma_det')
        self._logger.register_key('Train/max_kl_sigma')
        self._logger.register_key('Train/max_kl_mu')
        self._logger.register_key('Train/eta')

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """
        Update cost critic using TD3 algorithm.

        Args:
            obs (torch.Tensor): current observation
            act (torch.Tensor): current action
            cost (torch.Tensor): current cost
            done (torch.Tensor): current done signal
            next_obs (torch.Tensor): next observation

        Returns:
            None
        """
        with torch.no_grad():
            # Set the update noise and noise clip.
            self._actor_critic.target_actor.noise = self._cfgs.algo_cfgs.policy_noise
            next_action = self._actor_critic.target_actor.predict(next_obs, deterministic=False)
            next_q1_value_c, next_q2_value_c = self._actor_critic.target_cost_critic(
                next_obs, next_action
            )
            next_q_value_c = torch.min(next_q1_value_c, next_q2_value_c)
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c

        q1_value_c, q2_value_c = self._actor_critic.cost_critic(obs, action)
        loss = nn.functional.mse_loss(q1_value_c, target_q_value_c) + nn.functional.mse_loss(
            q2_value_c, target_q_value_c
        )

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(), self._cfgs.algo_cfgs.max_grad_norm
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()
        self._logger.store(
            **{
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q1_value_c.mean().item(),
            }
        )

    def _log_when_not_update(self) -> None:
        super()._log_when_not_update()
        self._logger.store(
            **{
                'Loss/Loss_l': 0.0,
                'Train/mean_sigma_det': 0.0,
                'Train/max_kl_sigma': 0.0,
                'Train/max_kl_mu': 0.0,
                'Train/eta': 0.0,
            }
        )
