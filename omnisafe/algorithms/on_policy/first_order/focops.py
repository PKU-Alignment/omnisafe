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

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed_utils


@registry.register
class FOCOPS(PolicyGradient, Lagrange):
    """The First Order Constrained Optimization in Policy Space (FOCOPS) algorithm.

    References:
        Paper Name: First Order Constrained Optimization in Policy Space.
        Paper author: Yiming Zhang, Quan Vuong, Keith W. Ross.
        Paper URL: https://arxiv.org/abs/2002.06506

    """

    def __init__(
        self,
        env_id,
        cfgs,
    ):
        r"""The :meth:`init` function."""

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
        self.algo = self.__class__.__name__
        self.lam = self.cfgs.lam
        self.eta = self.cfgs.eta

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/LagrangeMultiplier', self.lagrangian_multiplier)

    def compute_loss_pi(self, data: dict):
        """compute loss for policy"""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        kl_new_old = torch.distributions.kl.kl_divergence(dist, self.p_dist).sum(-1, keepdim=True)
        loss_pi = (
            kl_new_old
            - (1 / self.lam) * ratio * (data['adv'] - self.lagrangian_multiplier * data['cost_adv'])
        ) * (kl_new_old.detach() <= self.eta).type(torch.float32)
        loss_pi = loss_pi.mean()
        loss_pi -= self.cfgs.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = 0.5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """Update."""
        raw_data, data = self.buf.pre_process_data()
        # First update Lagrange multiplier parameter
        Jc = self.logger.get_stats('Metrics/EpCost')[0]
        self.update_lagrange_multiplier(Jc)
        # Then update policy network
        self.update_policy_net(data=data)
        # Update value network
        self.update_value_net(data=data)
        # Update cost network
        self.update_cost_net(data=data)
        return raw_data, data

    def slice_data(self, data) -> dict:
        """slice data for mini batch update"""

        slice_data = []
        obs = data['obs']
        act = data['act']
        target_v = data['target_v']
        log_p = data['log_p']
        adv = data['adv']
        discounted_ret = data['discounted_ret']
        cost_adv = data['cost_adv']
        target_v = data['target_v']
        batch_size = self.cfgs.batch_size
        for i in range(int(len(obs) / batch_size)):
            slice_data.append(
                {
                    'obs': obs[i * batch_size : (i + 1) * batch_size],
                    'act': act[i * batch_size : (i + 1) * batch_size],
                    'target_v': target_v[i * batch_size : (i + 1) * batch_size],
                    'log_p': log_p[i * batch_size : (i + 1) * batch_size],
                    'adv': adv[i * batch_size : (i + 1) * batch_size],
                    'discounted_ret': discounted_ret[i * batch_size : (i + 1) * batch_size],
                    'cost_adv': cost_adv[i * batch_size : (i + 1) * batch_size],
                }
            )

        return slice_data

    def update_policy_net(self, data) -> None:
        """update policy network"""

        # Slice data for mini batch update
        slice_data = self.slice_data(data)

        # Get prob. distribution before updates: used to measure KL distance
        with torch.no_grad():
            self.p_dist = self.actor_critic.actor(slice_data[0]['obs'])
        # Get loss and info values before update
        pi_l_old, _ = self.compute_loss_pi(data=slice_data[0])
        loss_pi_before = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.cfgs.actor_iters):

            for batch_data in slice_data:
                # Update policy network with batch data
                with torch.no_grad():
                    self.p_dist = self.actor_critic.actor(batch_data['obs'])

                self.actor_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data=batch_data)
                loss_pi.backward()
                # Apply L2 norm
                if self.cfgs.use_max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_critic.actor.parameters(), self.cfgs.max_grad_norm
                    )

                # Average grads across MPI processes
                distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
                self.actor_optimizer.step()

                q_dist = self.actor_critic.actor(batch_data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()

                if self.cfgs.kl_early_stopping:
                    # Average KL for consistent early stopping across processes
                    if distributed_utils.mpi_avg(torch_kl) > self.cfgs.target_kl:
                        self.logger.log(f'Reached ES criterion after {i+1} steps.')
                        break

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(
            **{
                'Loss/Loss_pi': loss_pi.item(),
                'Loss/Delta_loss_pi': loss_pi.item() - loss_pi_before,
                'Train/StopIter': i + 1,
                'Values/Adv': data['adv'].numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': torch_kl,
                'Train/PolicyRatio': pi_info['ratio'],
            }
        )
