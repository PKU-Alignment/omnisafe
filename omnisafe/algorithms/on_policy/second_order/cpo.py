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

from typing import NamedTuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class CPO(TRPO):
    """The Constrained Policy Optimization (CPO) algorithm.

    CPO is a derivative of TRPO.

    References:
        - Title: Constrained Policy Optimization
        - Authors: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel.
        - URL: https://arxiv.org/abs/1705.10528
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize CPO.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)
        self.cost_limit = cfgs.cost_limit
        self.loss_pi_cost_before = 0.0

    # pylint: disable-next=too-many-arguments,too-many-locals
    def search_step_size(
        self,
        step_dir: torch.Tensor,
        g_flat: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
        loss_pi_before: float,
        total_steps: int = 15,
        decay: float = 0.8,
        c: int = 0,
        optim_case: int = 0,
    ) -> tuple((torch.Tensor, int)):
        r"""Use line-search to find the step size that satisfies the constraint.

        CPO uses line-search to find the step size that satisfies the constraint.
        The constraint is defined as:

        .. math::
            J^C(\theta + \alpha \delta) - J^C(\theta) \leq \max \{0, c\}\\
            D_{KL}(\pi_{\theta}(\cdot|s) || \pi_{\theta + \alpha \delta}(\cdot|s)) \leq \delta_{KL}

        where :math:`\delta_{KL}` is the constraint of KL divergence, :math:`\alpha` is the step size,
        :math:`c` is the violation of constraint.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            cost_adv (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
            c (int, optional): The violation of constraint. Defaults to 0.
            optim_case (int, optional): The optimization case. Defaults to 0.
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
                loss_pi_rew, _ = self.compute_loss_pi(
                    obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
                )
                # Loss of cost of policy cost from real/expected reward
                loss_pi_cost, _ = self.compute_loss_cost_performance(
                    obs=obs, act=act, log_p=log_p, cost_adv=cost_adv
                )
                # Compute KL distance between new and old policy
                q_dist = self.actor_critic.actor(obs)
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
        r"""Log the CPO specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Misc/cost_gradient_norm
                -   The norm of the cost gradient.
            *   -   Misc/q
                -   The :math:`q` vector, which is the conjugate of Hessian :math:`H`.
            *   -   Misc/r
                -   The :math:`r` vector, where :math:`r = g^T H^{-1} b`.
            *   -   Misc/s
                -   The :math:`s` vector, where :math:`s = b^T H^{-1} b`
            *   -   Misc/A
                -   The A matrix, where :math:`A = q - \frac{r^2}{s}`
            *   -   Misc/B
                -   The B matrix, where :math:`B = 2 \delta_{KL} - \frac{c^2}{s}` ,
                    where :math:`c` is the cost violation in current epoch, and
                    :math:`\delta_{KL}` is the target KL divergence.
            *   -   Misc/Lambda_star
                -   The :math:`\lambda^*` vector.
            *   -   Misc/Nu_star
                -   The :math:`\nu^*` vector.
            *   -   Misc/OptimCase
                -   The optimization case.
        """
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

    def compute_loss_cost_performance(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        cost_adv: torch.Tensor,
    ):
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the loss of cost of policy cost from real cost.

        .. math::
            L = \mathbb{E}_{\pi} \left[ \frac{\pi(a|s)}{\pi_{old}(a|s)} A^C(s, a) \right]

        where :math:`A^C(s, a)` is the cost advantage,
        :math:`\pi_{old}(a|s)` is the old policy,
        :math:`\pi(a|s)` is the current policy.

        Args:
            obs (torch.Tensor): Observation.
            act (torch.Tensor): Action.
            log_p (torch.Tensor): Log probability.
            cost_adv (torch.Tensor): Cost advantage.
        """
        _, _log_p = self.actor_critic.actor(obs, act)
        ratio = torch.exp(_log_p - log_p)
        cost_loss = (ratio * cost_adv).mean()
        info = {}
        return cost_loss, info

    def update(self):
        """Update actor, critic, running statistics as we used in the :class:`PolicyGradient`.

        .. note::
            An additional step is to set the ``fvp_obs`` attribute to the current observation.
            The ``fvp_obs`` is the sub-sampled observation,
            which is used to accelerate the calculating of the Hessian-vector product.
        """
        raw_data, data = self.buf.pre_process_data()
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Update Policy Network
        obs, act, target_v, target_c, log_p, adv, cost_adv = (
            data['obs'],
            data['act'],
            data['target_v'],
            data['target_c'],
            data['log_p'],
            data['adv'],
            data['cost_adv'],
        )
        # Update critic
        self.update_value_net(obs=obs, target_v=target_v)
        if self.cfgs.use_cost:
            self.update_cost_net(obs=obs, target_c=target_c)
        # Update actor
        self.update_policy_net(obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv)
        return raw_data, data

    # pylint: disable=too-many-statements,too-many-locals,invalid-name,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        """Update policy network.

        Constrained Policy Optimization updates policy network using the conjugate gradient algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint. (Both KL divergence and cost limit).
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            cost_adv (torch.Tensor): The cost advantage tensor.
        """
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.actor_critic.actor.net)
        self.actor_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(
            obs=obs, act=act, log_p=log_p, adv=adv, cost_adv=cost_adv
        )
        loss_pi_before = loss_pi.item()
        # Get prob. distribution before updates, previous dist of possibilities
        p_dist = self.actor_critic.actor(obs)
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
        loss_cost, _ = self.compute_loss_cost_performance(
            obs=obs, act=act, log_p=log_p, cost_adv=cost_adv
        )
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
            elif cost < 0 and B >= 0:  # pylint: disable=chained-comparison
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
            obs=obs,
            act=act,
            log_p=log_p,
            adv=adv,
            cost_adv=cost_adv,
            total_steps=20,
        )
        # Update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor.net, new_theta)
        # Output the performance of pi policy on observation
        q_dist = self.actor_critic.actor(obs)
        torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()

        self.logger.store(
            **{
                'Values/Adv': act.numpy(),
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
