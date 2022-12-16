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
"""Implementation of the CPO algorithm."""

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.trpo import TRPO
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class CPO(TRPO):
    """The Constrained Policy Optimization (CPO) Algorithm.

    References:
        Paper Name: Constrained Policy Optimization.
        Paper author: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        Paper URL: https://arxiv.org/abs/1705.10528

    """

    def __init__(
        self,
        env,
        cfgs,
        algo='CPO',
    ):
        super().__init__(
            env=env,
            cfgs=cfgs,
            algo=algo,
        )
        self.cost_limit = cfgs.cost_limit
        self.loss_pi_cost_before = 0.0

    # pylint: disable-next=too-many-arguments,too-many-locals
    def search_step_size(
        self,
        step_dir,
        g_flat,
        p_dist,
        data,
        loss_pi_before,
        total_steps=25,
        decay=0.8,
        c=0,
        optim_case=0,
    ):
        r"""Use line-search to find the step size that satisfies the constraint.

        Args:
            step_dir
                direction theta changes towards
            g_flat
                gradient tensor of reward ,informs about how rewards improve with change of step direction
            c
                how much episode cost goes above limit
            p_dist
                inform about old policy, how the old policy p performs on observation this moment
            optim_case
                the way to optimize
            data
                data buffer,mainly with adv, costs, values, actions, and observations
            decay
                how search-step reduces in line-search
        """
        # Get distance each time theta goes towards certain direction
        step_frac = 1.0
        # Get and flatten parameters from pi-net
        _theta_old = get_flat_params_from(self.actor_critic.actor.net)
        # Reward improvement, g-flat as gradient of reward
        expected_rew_improve = g_flat.dot(step_dir)

        # While not within_trust_region and not finish all steps:
        for j in range(total_steps):
            # Get new theta
            new_theta = _theta_old + step_frac * step_dir
            # Set new theta as new actor parameters
            set_param_values_to_model(self.actor_critic.actor.net, new_theta)
            # The last acceptance steps to next step
            acceptance_step = j + 1

            with torch.no_grad():
                # Loss of policy reward from target/expected reward
                loss_pi_rew, _ = self.compute_loss_pi(data=data)
                # Loss of cost of policy cost from real/expected reward
                loss_pi_cost, _ = self.compute_loss_cost_performance(data=data)
                # Compute KL distance between new and old policy
                q_dist = self.actor_critic.actor(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            # Compute improvement of reward
            loss_rew_improve = loss_pi_before - loss_pi_rew.item()
            cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            # Average across MPI processes...
            torch_kl = distributed_utils.mpi_avg(torch_kl)
            # Pi_average of torch_kl above
            loss_rew_improve = distributed_utils.mpi_avg(loss_rew_improve)
            cost_diff = distributed_utils.mpi_avg(cost_diff)
            menu = (expected_rew_improve, loss_rew_improve)
            self.logger.log(f'Expected Improvement: {menu[0]} Actual: {menu[1]}')
            # Check whether there are nan.
            if not torch.isfinite(loss_pi_rew) and not torch.isfinite(loss_pi_cost):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_rew_improve < 0 if optim_case > 1 else False:
                self.logger.log('INFO: did not improve improve <0')
            # Change of cost's range
            elif cost_diff > max(-c, 0):
                self.logger.log(f'INFO: no improve {cost_diff} > {max(-c, 0)}')
            # Check KL-distance to avoid too far gap
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log(f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self.logger.log(f'Accept step at i={j + 1}')
                break
            step_frac *= decay
        else:
            # If didn't find a step satisfy those conditions
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.actor_critic.actor.net, _theta_old)
        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self):
        # Sign up for chosen log items
        TRPO.algorithm_specific_logs(self)
        self.logger.log_tabular('Misc/cost_gradient_norm')
        self.logger.log_tabular('Misc/A')
        self.logger.log_tabular('Misc/B')
        self.logger.log_tabular('Misc/q')
        self.logger.log_tabular('Misc/r')
        self.logger.log_tabular('Misc/s')
        self.logger.log_tabular('Misc/Lambda_star')
        self.logger.log_tabular('Misc/Nu_star')
        self.logger.log_tabular('Misc/OptimCase')

    def compute_loss_cost_performance(self, data):
        """
        Performance of cost on this moment
        """
        _, _log_p = self.actor_critic.actor(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        cost_loss = (ratio * data['cost_adv']).mean()
        info = {}
        return cost_loss, info

    def update_policy_net(
        self,
        data,
    ):  # pylint: disable=too-many-statements,too-many-locals,invalid-name
        """update policy net"""
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.actor_critic.actor.net)
        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        loss_pi_before = loss_pi.item()
        # Get prob. distribution before updates, previous dist of possibilities
        p_dist = self.actor_critic.actor(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # Average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
        g_flat = get_flat_gradients_from(self.actor_critic.actor.net)

        # Flip sign since policy_loss = -(ration * adv)
        g_flat *= -1
        # x: g or g_T in original paper, stands for gradient of cost function
        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)  # pylint: disable=invalid-name
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        # equivalent to : g^T x
        xHx = torch.dot(x, self.Fvp(x))
        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))
        assert xHx.item() >= 0, 'No negative values'

        # get the policy cost performance gradient b (flat as vector)
        self.actor_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor.net)
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = get_flat_gradients_from(self.actor_critic.actor.net)
        # :param ep_costs: do samplings to get approximate costs as ep_costs
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # :params c: how much sampled result of cost goes beyond limit
        cost = ep_costs - self.cost_limit
        # Rescale, and add small float to avoid nan
        cost /= self.logger.get_stats('Metrics/EpLen')[0] + eps  # rescale

        # Set variable names as used in the paper with conjugate_gradient method,
        # used to solve equation(compute Hessian Matrix) instead of Natural Gradient

        p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx  # conjugate of matrix H
        r = g_flat.dot(p)  # g^T H^{-1} b
        s = b_flat.dot(p)  # b^T H^{-1} b

        # optim_case: divided into 5 kinds to compute
        if b_flat.dot(b_flat) <= 1e-6 and cost < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all()
            assert torch.isfinite(s).all()

            # A,b: mathematical value, not too much true meaning
            A = q - r**2 / s  # must be always >= 0 (Cauchy-Schwarz inequality)
            B = 2 * self.target_kl - cost**2 / s  # safety line intersects trust-region if B > 0

            if cost < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif cost < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif cost >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                self.logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety half space is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self.logger.log('Alert! Attempting infeasible recovery!', 'red')

        # the following computes required nu_star and lambda_star
        if optim_case in [3, 4]:
            # under 3 and 4 cases directly use TRPO method
            alpha = torch.sqrt(
                2 * self.target_kl / (xHx + 1e-8)
            )  # step gap fixed by KKT condition in conjugate algorithm
            nu_star = torch.zeros(1)
            lambda_star = 1 / alpha
            step_dir = alpha * x  # change step direction to gap * gradient

        elif optim_case in [1, 2]:
            # in 1 and 2,
            def project_on_set(data: torch.Tensor, low: float, high: float) -> torch.Tensor:
                return torch.Tensor([max(low, min(data, high))])

            #  Analytical Solution to LQCLP, employ lambda,nu to compute final solution of OLOLQC
            #  λ=argmax(f_a(λ),f_b(λ)) = λa_star or λb_star
            #  computing formula shown in appendix, lambda_a and lambda_b
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self.target_kl))
            # λa_star = Proj(lambda_a ,0 ~ r/c)  λb_star=Proj(lambda_b,r/c~ +inf)
            # where projection(str,b,c)=max(b,min(str,c))
            # may be regarded as a projection from effective region towards safety region
            if cost < 0:
                lambda_a_star = project_on_set(lambda_a, 0.0, r / cost)
                lambda_b_star = project_on_set(lambda_b, r / cost, np.inf)
            else:
                lambda_a_star = project_on_set(lambda_a, r / cost, np.inf)
                lambda_b_star = project_on_set(lambda_b, 0.0, r / cost)

            def f_a(lam):
                return -0.5 * (A / (lam + eps) + B * lam) - r * cost / (s + eps)

            def f_b(lam):
                return -0.5 * (q / (lam + eps) + 2 * self.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )

            # Discard all negative values with torch.clamp(x, min=0)
            # Nu_star = (lambda_star * - r)/s
            nu_star = torch.clamp(lambda_star * cost - r, min=0) / (s + eps)
            # final x_star as final direction played as policy's loss to backward and update
            step_dir = 1.0 / (lambda_star + eps) * (x - nu_star * p)

        else:  # case == 0
            # purely decrease costs
            # without further check
            lambda_star = torch.zeros(1)
            nu_star = np.sqrt(2 * self.target_kl / (s + eps))
            step_dir = -nu_star * p

        final_step_dir, accept_step = self.search_step_size(
            step_dir,
            g_flat,
            c=cost,
            loss_pi_before=loss_pi_before,
            optim_case=optim_case,
            p_dist=p_dist,
            data=data,
            total_steps=20,
        )
        # Update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor.net, new_theta)
        # Output the performance of pi policy on observation
        q_dist = self.actor_critic.actor(data['obs'])
        torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()

        self.logger.store(
            **{
                'Values/Adv': data['act'].numpy(),
                'Train/Entropy': pi_info['ent'],
                'Train/KL': torch_kl,
                'Train/PolicyRatio': pi_info['ratio'],
                'Loss/Loss_pi': loss_pi_before,
                'Loss/Delta_loss_pi': loss_pi.item() - loss_pi_before,
                'Loss/Loss_cost_critic': 0.0,
                'Loss/Delta_loss_cost_critic': 0.0,
                'Train/StopIter': 1,
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': final_step_dir.norm().numpy(),
                'Misc/xHx': xHx.numpy(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/cost_gradient_norm': torch.norm(b_flat).numpy(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            }
        )
