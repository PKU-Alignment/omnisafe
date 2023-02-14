# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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

from typing import Any, Union

import numpy as np
import torch
from scipy.optimize import minimize
from torch.distributions import MultivariateNormal
from torch.nn.utils import clip_grad_norm_

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils.math import gaussian_kl


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

    def __init__(
        self,
        env_id: str,
        cfgs,
    ) -> None:
        """Constrained Variational Policy Optimization.

        Args:
            env_id (str): Environment ID.
            cfgs (dict): Configuration dictionary.
        """

        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )
        self.eta = 0.1
        self.lam = 0.1
        self.alpha_mean = 0.0
        self.alpha_var = 0.0
        self.cost_limit = self.cfgs.cost_limit

    def _specific_init_logs(self):
        super()._specific_init_logs()
        self.logger.register_key('Loss/Loss_l')
        self.logger.register_key('Misc/mean_sigma_det')
        self.logger.register_key('Misc/max_kl_sigma')
        self.logger.register_key('Misc/max_kl_mu')
        self.logger.register_key('Misc/eta')

    # pylint: disable-next=too-many-locals
    def update_policy_net(self, obs) -> None:
        """Update policy network.

        Args:
            obs (torch.Tensor): observation.
        """
        num_action = self.cfgs.sample_action_num
        num_obs = obs.shape[0]
        act_dim = self.actor_critic.act_dim
        obs_dim = self.actor_critic.obs_shape[0]

        with torch.no_grad():
            # sample N actions per state
            b_mean, _, b_var = self.ac_targ.actor.predict(
                obs, deterministic=True, need_log_prob=True
            )
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
            """Dual function of the non-parametric variational."""
            beta, lam = val
            target_q_np_comb = target_q_np - lam * target_qc_np
            max_q = np.max(target_q_np_comb, 1)
            return (
                beta * self.cfgs.dual_constraint
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
            self.alpha_mean -= (
                self.cfgs.alpha_mean_scale * (self.cfgs.kl_mean_constraint - kl_mu).detach().item()
            )
            self.alpha_var -= (
                self.cfgs.alpha_var_scale * (self.cfgs.kl_var_constraint - kl_sigma).detach().item()
            )

            self.alpha_mean = np.clip(self.alpha_mean, 0.0, self.cfgs.alpha_mean_max)
            self.alpha_var = np.clip(self.alpha_var, 0.0, self.cfgs.alpha_var_max)
            self.actor_optimizer.zero_grad()
            loss_l = -(
                loss_p
                + self.alpha_mean * (self.cfgs.kl_mean_constraint - kl_mu)
                + self.alpha_var * (self.cfgs.kl_var_constraint - kl_sigma)
            )
            loss_l.backward()
            clip_grad_norm_(self.actor_critic.actor.parameters(), 0.01)
            self.actor_optimizer.step()
            self.logger.store(
                **{
                    'Loss/Loss_pi': loss_p.mean().item(),
                    'Loss/Loss_l': loss_l.mean().item(),
                    'Misc/mean_sigma_det': sigma_det.item(),
                    'Misc/max_kl_sigma': kl_sigma.item(),
                    'Misc/max_kl_mu': kl_mu.item(),
                    'Misc/eta': self.eta,
                }
            )

    def algorithm_specific_logs(self):
        """Log the CVPO specific information."""


# pylint: disable-next=too-many-branches,too-many-return-statements
def to_ndarray(item: Any, dtype: np.dtype = None) -> Union[np.ndarray, TypeError, None]:
    """This function is used to convert the data type to ndarray.

    Change `torch.Tensor`, sequence of scalars to ndarray, and keep other data types unchanged.

    .. note:
        Now supports item type: :obj:`torch.Tensor`,  :obj:`dict`, :obj:`list`, :obj:`tuple` and :obj:`None`

    Args:
        item (Any): item to be converted.
        dtype (np.dtype): data type of the output ndarray. Default to None.
    """

    if isinstance(item, dict):
        new_data = {}
        for key, value in item.items():
            new_data[key] = to_ndarray(value, dtype)
        return new_data

    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return None
        if hasattr(item, '_fields'):  # namedtuple
            return type(item)(*[to_ndarray(t, dtype) for t in item])
        new_data = []
        for data in item:
            new_data.append(to_ndarray(data, dtype))
        return new_data

    if isinstance(item, torch.Tensor):
        if item.device != 'cpu':
            item = item.detach().cpu()
        if dtype is None:
            return item.numpy()
        return item.numpy().astype(dtype)

    if isinstance(item, np.ndarray):
        if dtype is None:
            return item
        return item.astype(dtype)

    if np.isscalar(item):
        return np.array(item)

    if item is None:
        return None

    raise TypeError(f'not support item type: {item}')
