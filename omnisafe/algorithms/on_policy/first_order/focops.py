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
"""Implementation of the FOCOPS algorithm."""

from typing import Dict, Tuple

import torch
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.utils.config import Config


@registry.register
class FOCOPS(PolicyGradient):
    """The First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
    """

    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')

    def __init__(self, env_id: str, cfgs: Config) -> None:
        super().__init__(env_id, cfgs)
        self._p_dist: Normal

    def _loss_pi(
        self, obs: torch.Tensor, act: torch.Tensor, logp: torch.Tensor, adv: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)
        loss = (kl - (1 / self._cfgs.algo_cfgs.focops_lam) * ratio * adv) * (
            kl.detach() <= self._cfgs.algo_cfgs.focops_eta
        ).type(torch.float32)
        loss = loss.mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()

        entrophy = distribution.entropy().mean().item()
        info = {'entrophy': entrophy, 'ratio': ratio.mean().item(), 'std': std}
        return loss, info

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        return (adv_r - self._lagrange.lagrangian_multiplier * adv_c) / (
            1 + self._lagrange.lagrangian_multiplier
        )

    def _update(self) -> None:
        """Update actor, critic, running statistics as we used in the :class:`PolicyGradient`.

        In addition, we also update the Lagrange multiplier parameter,
        by calling the :meth:`update_lagrange_multiplier` function.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        original_obs = obs
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        dataloader = DataLoader(
            dataset=TensorDataset(
                obs, act, logp, target_value_r, target_value_c, adv_r, adv_c, old_mean, old_std
            ),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for i in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
                old_mean,
                old_std,
            ) in dataloader:
                self._update_rewrad_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)

                self._p_dist = Normal(old_mean, old_std)
                self._update_actor(obs, act, logp, adv_r, adv_c)

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Train/StopIter': i + 1,
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': kl,
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier,
            }
        )
