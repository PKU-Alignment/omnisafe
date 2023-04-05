# Copyright 2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the CUP algorithm."""

import torch
from rich.progress import track
from torch.distributions import Normal
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed
from omnisafe.utils.config import Config


@registry.register
class CUP(PPO):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
                    Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
    """

    def _init(self) -> None:
        super()._init()
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        r"""Log the CUP specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   ``Metrics/LagrangeMultiplier``
                -   The Lagrange multiplier.
            *   -   ``Train/MaxRatio``
                -   The maximum ratio between the current policy and the old policy.
            *   -   ``Train/MinRatio``
                -   The minimum ratio between the current policy and the old policy.
            *   -   ``Loss/Loss_pi_c``
                -   The loss of the cost performance.
            *   -   ``Train/SecondStepStopIter``
                -   The number of iterations to stop the second step.
            *   -   ``Train/SecondStepEntropy``
                -   The entropy of the current policy.
            *   -   ``Train/SecondStepPolicyRatio``
                -   The ratio between the current policy and the old policy.
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Train/MaxRatio')
        self._logger.register_key('Train/MinRatio')
        self._logger.register_key('Loss/Loss_pi_c', delta=True)
        self._logger.register_key('Train/SecondStepStopIter')
        self._logger.register_key('Train/SecondStepEntropy')
        self._logger.register_key('Train/SecondStepPolicyRatio')

    def __init__(self, env_id: str, cfgs: Config) -> None:
        super().__init__(env_id, cfgs)
        self._p_dist: Normal
        self._max_ratio: float = 0.0
        self._min_ratio: float = 0.0

    def _loss_pi_cost(self, obs, act, logp, adv_c):
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the KL divergence between the current policy and the old policy,
        the entropy of the current policy, and the ratio between the current policy and the old
        policy.

        The loss of the cost performance is defined as:

        .. math::
            L = \underset{a \sim \pi_{\theta}}{\mathbb{E}}[\lambda \frac{1 - \gamma \nu}{1 - \gamma}
            \frac{\pi_\theta^{'}(a|s)}{\pi_\theta(a|s)} A^{C}_{\pi_{\theta}}
            + KL(\pi_\theta^{'}(a|s)||\pi_\theta(a|s))]

        where :math:`\lambda` is the Lagrange multiplier,
        :math:`\frac{1 - \gamma \nu}{1 - \gamma}` is the coefficient value,
        :math:`\pi_\theta^{'}(a_t|s_t)` is the current policy,
        :math:`\pi_\theta(a_t|s_t)` is the old policy,
        :math:`A^{C}_{\pi_{\theta}}` is the cost advantage,
        :math:`KL(\pi_\theta^{'}(a_t|s_t)||\pi_\theta(a_t|s_t))` is the KL divergence between the
        current policy and the old policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            cost_adv (torch.Tensor): Cost advantage.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)

        kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)

        coef = (1 - self._cfgs.algo_cfgs.gamma * self._cfgs.algo_cfgs.lam) / (
            1 - self._cfgs.algo_cfgs.gamma
        )
        loss = (self._lagrange.lagrangian_multiplier * coef * ratio * adv_c + kl).mean()

        # useful extra info
        temp_max = torch.max(ratio).detach().mean().item()
        temp_min = torch.min(ratio).detach().mean().item()
        if temp_max > self._max_ratio:
            self._max_ratio = temp_max
        if temp_min < self._min_ratio:
            self._min_ratio = temp_min
        entropy = distribution.entropy().mean().item()
        info = {'entropy': entropy, 'ratio': ratio.mean().item(), 'std': std}

        self._logger.store(**{'Loss/Loss_pi_c': loss.item()})

        return loss, info

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters.

        In CUP, the Lagrange multiplier is updated as the naive lagrange multiplier update:

        .. math::
            \lambda_{k+1} = \lambda_k + \eta (J^{C}_{\pi_\theta} - C)

        where :math:`\lambda_k` is the Lagrange multiplier at iteration :math:`k`,
        :math:`\eta` is the Lagrange multiplier learning rate,
        :math:`J^{C}_{\pi_{\theta}}` is the cost of the current policy,
        and :math:`C` is the cost limit.

        Then in each iteration of the policy update, CUP calculates current policy's
        distribution, which used to calculate the policy loss.

        Args:
            self (object): object of the class.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

        super()._update()

        data = self._buf.get()
        obs, act, logp, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['adv_c'],
        )
        original_obs = obs
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, adv_c, old_mean, old_std),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for obs, act, logp, adv_c, old_mean, old_std in dataloader:
                self._p_dist = Normal(old_mean, old_std)
                loss_cost, info = self._loss_pi_cost(obs, act, logp, adv_c)
                self._actor_critic.actor_optimizer.zero_grad()
                loss_cost.backward()
                if self._cfgs.algo_cfgs.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._actor_critic.actor.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self._actor_critic.actor)
                self._actor_critic.actor_optimizer.step()

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            **{
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.item(),
                'Train/MaxRatio': self._max_ratio,
                'Train/MinRatio': self._min_ratio,
                'Train/SecondStepStopIter': i + 1,  # pylint: disable=undefined-loop-variable
                'Train/SecondStepEntropy': info['entropy'],
                'Train/SecondStepPolicyRatio': info['ratio'],
            },
        )
