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
"""Implementation of the CUP algorithm."""

from typing import NamedTuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed_utils


@registry.register
class CUP(PPO, Lagrange):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
                 Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize CUP.

        CUP is a combination of :class:`PPO` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        PPO.__init__(
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
            lagrangian_upper_bound=self.cfgs.lagrange_cfgs.lagrangian_upper_bound,
        )
        self.lam = self.cfgs.lam
        self.eta = self.cfgs.eta
        self.max_ratio = 0
        self.min_ratio = 0
        self.p_dist_batch = None

    def algorithm_specific_logs(self) -> None:
        """Log the CUP specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
            *   -   Train/MaxRatio
                -   The maximum ratio between the current policy and the old policy.
            *   -   Train/MinRatio
                -   The minimum ratio between the current policy and the old policy.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())
        self.logger.log_tabular('Train/MaxRatio', self.max_ratio)
        self.logger.log_tabular('Train/MinRatio', self.min_ratio)

    # pylint: disable=too-many-locals
    def compute_loss_cost_performance(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        cost_adv: torch.Tensor,
    ):
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the KL divergence between the current policy and the old policy,
        the entropy of the current policy, and the ratio between the current policy and the old
        policy.

        The loss of the cost performance is defined as:

        .. math::
            L = \underset{a \sim \pi_{\theta^{old}}}{\mathbb{E}}[\lambda \frac{1 - \gamma \nu}{1 - \gamma}
            \frac{\pi_\theta(a|s)}{\pi_\theta^{old}(a|s)} A^{C}_{\pi_{\theta}^{old}}
            + KL(\pi_\theta(a|s)||\pi_\theta^{old}(a|s))]

        where :math:`\lambda` is the Lagrange multiplier,
        :math:`\frac{1 - \gamma \nu}{1 - \gamma}` is the coefficient value,
        :math:`\pi_\theta(a_t|s_t)` is the current policy,
        :math:`\pi_\theta^{old}(a_t|s_t)` is the old policy,
        :math:`A^{C}_{\pi_{\theta}^{old}}` is the cost advantage,
        :math:`KL(\pi_\theta(a_t|s_t)||\pi_\theta^{old}(a_t|s_t))` is the KL divergence between the
        current policy and the old policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            cost_adv (torch.Tensor): Cost advantage.
        """
        dist, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist_batch).sum(
            -1, keepdim=True
        )

        coef = (1 - self.cfgs.buffer_cfgs.gamma * self.cfgs.buffer_cfgs.lam) / (
            1 - self.cfgs.buffer_cfgs.gamma
        )
        cost_loss = (self.lagrangian_multiplier * coef * ratio * cost_adv + kl_new_old).mean()

        # Useful extra info
        temp_max = torch.max(ratio).detach().numpy()
        temp_min = torch.min(ratio).detach().numpy()
        if temp_max > self.max_ratio:
            self.max_ratio = temp_max
        if temp_min < self.min_ratio:
            self.min_ratio = temp_min
        approx_kl = 0.5 * (log_p - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return cost_loss, pi_info

    def update(self):
        """Update actor, critic, running statistics as we used in the :class:`PPO`.

        .. note::

            An additional step is added to update the Lagrange multiplier,
            which is defined in the :meth:`update_lagrange_multiplier` function.

        """
        raw_data, data = self.buf.pre_process_data()
        obs, act, target_v, target_c, log_p, adv, cost_adv = (
            data['obs'],
            data['act'],
            data['target_v'],
            data['target_c'],
            data['log_p'],
            data['adv'],
            data['cost_adv'],
        )
        # First update Lagrange multiplier parameter
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        self.update_lagrange_multiplier(Jc)
        # Then update value network
        self.update_value_net(obs=obs, target_v=target_v)
        self.update_cost_net(obs=obs, target_c=target_c)
        # Update actor
        self.update_policy_net(obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv)
        return raw_data, data

    # pylint: disable=too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        r"""Update policy network.

        CUP first performs a number of gradient descent steps to maximize reward,
        and then performs a number of gradient descent steps to minimize cost.

        The pseudo code is shown below:

        .. code-block:: python

            for _ in range(self.cfgs.actor_iters):
                for _ in range(self.cfgs.num_mini_batches):
                    # Get mini-batch data
                    # Compute loss of reward performance
                    # Update network
                for _ in range(self.cfgs.num_mini_batches):
                    # Get mini-batch data
                    # Compute loss of cost performance
                    # Update network

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            adv (torch.Tensor): Advantage.
            cost_adv (torch.Tensor): Cost advantage.
        """
        # Divide whole local epoch data into mini_batches
        mbs = self.local_steps_per_epoch // self.cfgs.num_mini_batches
        assert mbs >= 16, f'Batch size {mbs}<16'

        with torch.no_grad():
            self.p_dist = self.actor_critic.actor(obs)

        # Get loss and info values before update
        pi_l_old, _ = self.compute_loss_pi(
            obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
        )
        loss_pi_before = pi_l_old.item()
        indices = np.arange(self.local_steps_per_epoch)
        pi_loss = []
        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs.actor_iters):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.actor_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(
                    obs=obs[mb_indices],
                    act=act[mb_indices],
                    log_p=log_p[mb_indices],
                    adv=adv[mb_indices],
                    cost_adv=cost_adv[mb_indices],
                )
                loss_pi.backward()
                pi_loss.append(loss_pi.item())
                # Apply L2 norm
                if self.cfgs.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
                    )

                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
                self.actor_optimizer.step()

            q_dist = self.actor_critic.actor(obs)
            torch_kl = torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()

            if self.cfgs.kl_early_stopping:
                # Average KL for consistent early stopping across processes
                if distributed_utils.mpi_avg(torch_kl) > self.cfgs.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break
        with torch.no_grad():
            self.p_dist = self.actor_critic.actor(obs)
        # Second, CUP perform a number of gradient descent steps to minimize cost
        for i in range(self.cfgs.actor_iters):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.local_steps_per_epoch, mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.actor_optimizer.zero_grad()
                with torch.no_grad():
                    self.p_dist_batch = self.actor_critic.actor(obs[mb_indices])
                loss_pi, pi_info = self.compute_loss_cost_performance(
                    obs=obs[mb_indices],
                    act=act[mb_indices],
                    log_p=log_p[mb_indices],
                    cost_adv=cost_adv[mb_indices],
                )
                loss_pi.backward()
                pi_loss.append(loss_pi.item())
                # Apply L2 norm
                if self.cfgs.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
                    )

                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
                self.actor_optimizer.step()

            q_dist = self.actor_critic.actor(obs)
            torch_kl = torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()

            if self.cfgs.kl_early_stopping:
                # Average KL for consistent early stopping across processes
                if distributed_utils.mpi_avg(torch_kl) > self.cfgs.target_kl:
                    self.logger.log(f'Reached ES criterion after {i+1} steps.')
                    break

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(
            **{
                'Loss/Loss_pi': np.mean(pi_loss),
                'Loss/Delta_loss_pi': np.mean(pi_loss) - loss_pi_before,
                'Train/StopIter': i + 1,
                'Values/Adv': adv.numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': torch_kl,
                'Train/PolicyRatio': pi_info['ratio'],
            }
        )
