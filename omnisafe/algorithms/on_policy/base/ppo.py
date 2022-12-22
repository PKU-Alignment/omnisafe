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
"""Implementation of the PPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.utils import distributed_utils


@registry.register
class PPO(PolicyGradient):
    """The Proximal Policy Optimization Algorithms (PPO) Algorithm.

    References:
        Paper Name: Proximal Policy Optimization Algorithms.
        Paper author: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        Paper URL: https://arxiv.org/pdf/1707.06347.pdf
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env_id,
        cfgs,
    ):
        """Initialize PPO."""
        self.clip = cfgs.clip
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )

    def compute_loss_pi(self, data: dict):
        """Compute policy loss."""
        dist, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        # Importance ratio
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        loss_pi += self.cfgs.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

        return loss_pi, pi_info

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
