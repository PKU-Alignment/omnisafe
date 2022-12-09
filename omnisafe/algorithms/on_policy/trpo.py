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
"""Implementation of the TRPO algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.natural_pg import NaturalPG
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class TRPO(NaturalPG):
    """Class for TRPO."""

    def __init__(
        self,
        env,
        cfgs,
        algo='TRPO',
    ):
        NaturalPG.__init__(
            self,
            env=env,
            cfgs=cfgs,
            algo=algo,
        )

    # pylint: disable-next=too-many-arguments, too-many-locals, arguments-differ
    def search_step_size(
        self,
        step_dir,
        g_flat,
        p_dist,
        data,
        loss_pi_before,
        total_steps=15,
        decay=0.8,
    ):
        """
        TRPO performs line-search until constraint satisfaction.
        main idea: search around for a satisfied step of policy update to improve loss and reward performance
        :param step_dir:direction theta changes towards
        :param g_flat:  gradient tensor of reward ,informs about how rewards improve with change of step direction
        :param c:how much episode cost goes above limit
        :param p_dist: inform about old policy, how the old policy p performs on observation this moment
        :param optim_case: the way to optimize
        :param data: data buffer,mainly with adv, costs, values, actions, and observations
        :param decay: how search-step reduces in line-search
        """
        # How far to go in a single update
        step_frac = 1.0
        # Get old parameterized policy expression
        _theta_old = get_flat_params_from(self.actor_critic.actor.net)
        # Change expected objective function gradient = expected_imrpove best this moment
        expected_improve = g_flat.dot(step_dir)

        # While not within_trust_region and not out of total_steps:
        for j in range(total_steps):
            # Update theta params
            new_theta = _theta_old + step_frac * step_dir
            # Set new params as params of net
            set_param_values_to_model(self.actor_critic.actor.net, new_theta)
            # The stepNo this update accept
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, _ = self.compute_loss_pi(data=data)
                # Compute KL distance between new and old policy
                q_dist = self.actor_critic.actor(data['obs'])
                # KL-distance of old p-dist and new q-dist, applied in KLEarlyStopping
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            # Real loss improve: old policy loss - new policy loss
            loss_improve = loss_pi_before - loss_pi.item()
            # Average processes.... multi-processing style like: mpi_tools.mpi_avg(xxx)
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            loss_improve = distributed_utils.mpi_avg(loss_improve)
            menu = (expected_improve, loss_improve)
            self.logger.log(f'Expected Improvement: {menu[0]} Actual: {menu[1]}')
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.actor_critic.actor.net, _theta_old)

        return step_frac * step_dir, acceptance_step

    # pylint: disable-next=too-many-locals
    def update_policy_net(self, data):
        """update policy network"""
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.actor_critic.actor.net)
        self.actor_critic.actor.net.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        loss_pi_before = distributed_utils.mpi_avg(loss_pi.item())
        p_dist = self.actor_critic.actor(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
        g_flat = get_flat_gradients_from(self.actor_critic.actor.net)
        g_flat *= -1

        # pylint: disable-next=invalid-name
        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        # pylint: disable-next=invalid-name
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.search_step_size(
            step_dir=step_direction,
            g_flat=g_flat,
            p_dist=p_dist,
            loss_pi_before=loss_pi_before,
            data=data,
        )

        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor.net, new_theta)

        with torch.no_grad():
            q_dist = self.actor_critic.actor(data['obs'])
            # pylint: disable-next=invalid-name
            kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_pi, pi_info = self.compute_loss_pi(data=data)

        self.logger.store(
            **{
                'Values/Adv': data['adv'].numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': kl,
                'Train/PolicyRatio': pi_info['ratio'],
                'Train/StopIter': 1,
                'Loss/Loss_pi': loss_pi.item(),
                'Loss/Delta_loss_pi': loss_pi.item() - loss_pi_before,
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(final_step_dir).numpy(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )
