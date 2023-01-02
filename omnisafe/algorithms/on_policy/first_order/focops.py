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
"""Implementation of the FOCOPS algorithm."""

from typing import NamedTuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed_utils


@registry.register
class FOCOPS(PolicyGradient, Lagrange):
    """The First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        - Title: First Order Constrained Optimization in Policy Space
        - Authors: Yiming Zhang, Quan Vuong, Keith W. Ross.
        - URL: `FOCOPS <https://arxiv.org/abs/2002.06506>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize FOCOPS.

        FOCOPS is a combination of :class:`PolicyGradient` and :class:`Lagrange` model.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        PolicyGradient.__init__(
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
        self.p_dist_batch = None

    def algorithm_specific_logs(self) -> None:
        """Log the FOCOPS specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/LagrangeMultiplier
                -   The Lagrange multiplier value in current epoch.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier.item())

    # pylint: disable=too-many-arguments
    def compute_loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> tuple((torch.Tensor, dict)):
        r"""
        Computing pi/actor loss.
        In FOCOPS, the loss is defined as:

        .. math::
            :nowrap:

            \begin{eqnarray}
            L = \nabla_\theta D_{K L}\left(\pi_\theta \| \pi_{\theta^{old}}\right)[s]
            -\frac{1}{\eta} \underset{a \sim \pi_{\theta^{old}}}
            {\mathbb{E}}\left[\frac{\nabla_\theta \pi_\theta(a \mid s)}
            {\pi_{\theta^{old}}(a \mid s)}\left(A^{R}_{\pi_{\theta^{old}}}(s, a)
            -\lambda A^C_{\pi_{\theta^{old}}}(s, a)\right)\right]
            \end{eqnarray}

        where :math:`\eta` is a hyperparameter, :math:`\lambda` is the Lagrange multiplier,
        :math:`A_{\pi_{\theta_k}}(s, a)` is the advantage function,
        :math:`A^C_{\pi_{\theta_k}}(s, a)` is the cost advantage function,
        :math:`\pi^*` is the optimal policy, and :math:`\pi_{\theta_k}` is the current policy.
        """
        dist, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)
        loss_pi = (
            kl_new_old - (1 / self.lam) * ratio * (adv - self.lagrangian_multiplier * cost_adv)
        ) * (kl_new_old.detach() <= self.eta).type(torch.float32)
        loss_pi = loss_pi.mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = 0.5 * (log_p - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self) -> None:
        """Update actor, critic, running statistics as we used in the :class:`PolicyGradient`.

        In addition, we also update the Lagrange multiplier parameter,
        by calling the :meth:`update_lagrange_multiplier` function.
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

    # pylint: disable=too-many-locals
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        r"""Update policy network under a double for loop.

            The pseudo code is shown below:

            .. code-block:: python

                for _ in range(self.cfgs.actor_iters):
                    for _ in range(self.cfgs.num_mini_batches):
                        # Get mini-batch data
                        # Compute loss
                        # Update network

            .. warning::
                For some ``KL divergence`` based algorithms (e.g. TRPO, CPO, etc.),
                the ``KL divergence`` between the old policy and the new policy is calculated.
                And the ``KL divergence`` is used to determine whether the update is successful.
                If the ``KL divergence`` is too large, the update will be terminated.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log_p`` stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
            cost_adv (torch.Tensor): ``cost_advantage`` stored in buffer.
        """
        # Divide whole local epoch data into mini_batches
        mbs = self.local_steps_per_epoch // self.cfgs.num_mini_batches
        assert mbs >= 16, f'Batch size {mbs}<16'

        with torch.no_grad():
            self.p_dist = self.actor_critic.actor(obs)
            old_dist = self.p_dist

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
                with torch.no_grad():
                    self.p_dist = self.actor_critic.actor(obs[mb_indices])
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
            torch_kl = torch.distributions.kl.kl_divergence(old_dist, q_dist).mean().item()

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
