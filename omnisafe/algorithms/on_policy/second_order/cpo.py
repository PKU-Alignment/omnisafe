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
"""Implementation of the CPO algorithm."""

from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
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
        - URL: `CPO <https://arxiv.org/abs/1705.10528>`_
    """

    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Misc/cost_gradient_norm')
        self._logger.register_key('Misc/A')
        self._logger.register_key('Misc/B')
        self._logger.register_key('Misc/q')
        self._logger.register_key('Misc/r')
        self._logger.register_key('Misc/s')
        self._logger.register_key('Misc/Lambda_star')
        self._logger.register_key('Misc/Nu_star')
        self._logger.register_key('Misc/OptimCase')

    # pylint: disable-next=too-many-arguments,too-many-locals
    def _cpo_search_step(
        self,
        step_direction: torch.Tensor,
        grads: torch.Tensor,
        p_dist: torch.distributions.Distribution,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
        loss_reward_before: torch.Tensor,
        loss_cost_before: torch.Tensor,
        total_steps: int = 15,
        decay: float = 0.8,
        violation_c: int = 0,
        optim_case: int = 0,
    ) -> tuple[torch.Tensor, int]:
        r"""Use line-search to find the step size that satisfies the constraint.

        CPO uses line-search to find the step size that satisfies the constraint. The constraint is
        defined as:

        .. math::

            J^C (\theta + \alpha \delta) - J^C (\theta) \leq \max \{ 0, c \} \\
            D_{KL} (\pi_{\theta} (\cdot|s) || \pi_{\theta + \alpha \delta} (\cdot|s)) \leq \delta_{KL}

        where :math:`\delta_{KL}` is the constraint of KL divergence, :math:`\alpha` is the step size,
        :math:`c` is the violation of constraint.

        Args:
            step_dir (torch.Tensor): The step direction.
            g_flat (torch.Tensor): The gradient of the policy.
            p_dist (torch.distributions.Distribution): The old policy distribution.
            obs (torch.Tensor): The observation.
            act (torch.Tensor): The action.
            logp (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage.
            adv_c (torch.Tensor): The cost advantage.
            loss_pi_before (float): The loss of the policy before the update.
            total_steps (int, optional): The total steps to search. Defaults to 15.
            decay (float, optional): The decay rate of the step size. Defaults to 0.8.
            violation_c (int, optional): The violation of constraint. Defaults to 0.
            optim_case (int, optional): The optimization case. Defaults to 0.

        Returns:
            A tuple of final step direction and the size of acceptance steps.
        """
        # get distance each time theta goes towards certain direction
        step_frac = 1.0
        # get and flatten parameters from pi-net
        theta_old = get_flat_params_from(self._actor_critic.actor)
        # reward improvement, g-flat as gradient of reward
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        # while not within_trust_region and not finish all steps:
        for step in range(total_steps):
            # get new theta
            new_theta = theta_old + step_frac * step_direction
            # set new theta as new actor parameters
            set_param_values_to_model(self._actor_critic.actor, new_theta)
            # the last acceptance steps to next step
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    # loss of policy reward from target/expected reward
                    loss_reward = self._loss_pi(obs=obs, act=act, logp=logp, adv=adv_r)
                except ValueError:
                    step_frac *= decay
                    continue
                # loss of cost of policy cost from real/expected reward
                loss_cost = self._loss_pi_cost(obs=obs, act=act, logp=logp, adv_c=adv_c)
                # compute KL distance between new and old policy
                q_dist = self._actor_critic.actor(obs)
                kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()
            # compute improvement of reward
            loss_reward_improve = loss_reward_before - loss_reward
            # compute difference of cost
            loss_cost_diff = loss_cost - loss_cost_before

            # average across MPI processes...
            kl = distributed.dist_avg(kl)
            # pi_average of torch_kl above
            loss_reward_improve = distributed.dist_avg(loss_reward_improve)
            loss_cost_diff = distributed.dist_avg(loss_cost_diff)
            self._logger.log(
                f'Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}',
            )
            # check whether there are nan.
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                self._logger.log('WARNING: loss_pi not finite')
            if not torch.isfinite(kl):
                self._logger.log('WARNING: KL not finite')
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                self._logger.log('INFO: did not improve improve <0')
            # change of cost's range
            elif loss_cost_diff > max(-violation_c, 0):
                self._logger.log(f'INFO: no improve {loss_cost_diff} > {max(-violation_c, 0)}')
            # check KL-distance to avoid too far gap
            elif kl > self._cfgs.algo_cfgs.target_kl:
                self._logger.log(f'INFO: violated KL constraint {kl} at step {step + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self._logger.log(f'Accept step at i={step + 1}')
                break
            step_frac *= decay
        else:
            # if didn't find a step satisfy those conditions
            self._logger.log('INFO: no suitable step found...')
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        self._logger.store(
            {
                'Train/KL': kl,
            },
        )

        set_param_values_to_model(self._actor_critic.actor, theta_old)
        return step_frac * step_direction, acceptance_step

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the performance of cost on this moment.

        We compute the loss of cost of policy cost from real cost.

        .. math::

            L = \mathbb{E}_{\pi} \left[ \frac{\pi^{'} (a|s)}{\pi (a|s)} A^C (s, a) \right]

        where :math:`A^C (s, a)` is the cost advantage, :math:`\pi (a|s)` is the old policy,
        and :math:`\pi^{'} (a|s)` is the current policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The loss of the cost performance.
        """
        self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        return (ratio * adv_c).mean()

    # pylint: disable=invalid-name
    def _determine_case(
        self,
        b_grads: torch.Tensor,
        ep_costs: torch.Tensor,
        q: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor]:
        """Determine the case of the trust region update.

        Args:
            b_grad (torch.Tensor): Gradient of the cost function.
            ep_costs (torch.Tensor): Cost of the current episode.
            q (torch.Tensor): The quadratic term of the quadratic approximation of the cost function.
            r (torch.Tensor): The linear term of the quadratic approximation of the cost function.
            s (torch.Tensor): The constant term of the quadratic approximation of the cost function.

        Returns:
            optim_case: The case of the trust region update.
            A: The quadratic term of the quadratic approximation of the cost function.
            B: The linear term of the quadratic approximation of the cost function.
        """
        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), 'r is not finite'
            assert torch.isfinite(s).all(), 's is not finite'

            A = q - r**2 / (s + 1e-8)
            B = 2 * self._cfgs.algo_cfgs.target_kl - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif ep_costs < 0 <= B:
                # point in trust region is feasible but safety boundary intersects
                # ==> only part of trust region is feasible
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                # point in trust region is infeasible and cost boundary doesn't intersect
                # ==> entire trust region is infeasible
                optim_case = 1
                self._logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety half space is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self._logger.log('Alert! Attempting infeasible recovery!', 'red')

        return optim_case, A, B

    # pylint: disable=invalid-name, too-many-arguments, too-many-locals
    def _step_direction(
        self,
        optim_case: int,
        xHx: torch.Tensor,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        q: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        s: torch.Tensor,
        ep_costs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if optim_case in (3, 4):
            # under 3 and 4 cases directly use TRPO method
            alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(data: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
                """Project data to [low, high] interval."""
                return torch.clamp(data, low, high)

            #  analytical Solution to LQCLP, employ lambda,nu to compute final solution of OLOLQC
            #  λ=argmax(f_a(λ),f_b(λ)) = λa_star or λb_star
            #  computing formula shown in appendix, lambda_a and lambda_b
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self._cfgs.algo_cfgs.target_kl))
            # λa_star = Proj(lambda_a ,0 ~ r/c)  λb_star=Proj(lambda_b,r/c~ +inf)
            # where projection(str,b,c)=max(b,min(str,c))
            # may be regarded as a projection from effective region towards safety region
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(lambda_a, torch.as_tensor(0.0), r_num / eps_cost)
                lambda_b_star = project(lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf))
            else:
                lambda_a_star = project(lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf))
                lambda_b_star = project(lambda_b, torch.as_tensor(0.0), r_num / eps_cost)

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * self._cfgs.algo_cfgs.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )

            # discard all negative values with torch.clamp(x, min=0)
            # Nu_star = (lambda_star * - r)/s
            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)
            # final x_star as final direction played as policy's loss to backward and update
            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:  # case == 0
            # purely decrease costs
            # without further check
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (s + 1e-8))
            step_direction = -nu_star * p

        return step_direction, lambda_star, nu_star

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        """Update policy network.

        Constrained Policy Optimization updates policy network using the
        `conjugate gradient <https://en.wikipedia.org/wiki/Conjugate_gradient_method>`_ algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint.
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            logp (torch.Tensor): The log probability of the action.
            adv_r (torch.Tensor): The reward advantage tensor.
            adv_c (torch.Tensor): The cost advantage tensor.
        """
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        loss_reward = self._loss_pi(obs, act, logp, adv_r)
        loss_reward_before = distributed.dist_avg(loss_reward)
        p_dist = self._actor_critic.actor(obs)

        loss_reward.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grads = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grads, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = x.dot(self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))

        self._actor_critic.zero_grad()
        loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
        loss_cost_before = distributed.dist_avg(loss_cost)

        loss_cost.backward()
        distributed.avg_grads(self._actor_critic.actor)

        b_grads = get_flat_gradients_from(self._actor_critic.actor)
        ep_costs = self._logger.get_stats('Metrics/EpCost')[0] - self._cfgs.algo_cfgs.cost_limit

        p = conjugate_gradients(self._fvp, b_grads, self._cfgs.algo_cfgs.cg_iters)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        optim_case, A, B = self._determine_case(
            b_grads=b_grads,
            ep_costs=ep_costs,
            q=q,
            r=r,
            s=s,
        )

        step_direction, lambda_star, nu_star = self._step_direction(
            optim_case=optim_case,
            xHx=xHx,
            x=x,
            A=A,
            B=B,
            q=q,
            p=p,
            r=r,
            s=s,
            ep_costs=ep_costs,
        )

        step_direction, accept_step = self._cpo_search_step(
            step_direction=step_direction,
            grads=grads,
            p_dist=p_dist,
            obs=obs,
            act=act,
            logp=logp,
            adv_r=adv_r,
            adv_c=adv_c,
            loss_reward_before=loss_reward_before,
            loss_cost_before=loss_cost_before,
            total_steps=20,
            violation_c=ep_costs,
            optim_case=optim_case,
        )

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss_reward = self._loss_pi(obs, act, logp, adv_r)
            loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
            loss = loss_reward + loss_cost

        self._logger.store(
            {
                'Loss/Loss_pi': loss.item(),
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': step_direction.norm().mean().item(),
                'Misc/xHx': xHx.mean().item(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(grads).mean().item(),
                'Misc/cost_gradient_norm': torch.norm(b_grads).mean().item(),
                'Misc/Lambda_star': lambda_star.item(),
                'Misc/Nu_star': nu_star.item(),
                'Misc/OptimCase': int(optim_case),
                'Misc/A': A.item(),
                'Misc/B': B.item(),
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            },
        )
