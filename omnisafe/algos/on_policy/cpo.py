"""CPO"""
import numpy as np
import torch

import omnisafe.algos.utils.distributed_tools as distributed_tools
from omnisafe.algos.on_policy.trpo import TRPO
from omnisafe.algos.registry import REGISTRY
from omnisafe.algos.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@REGISTRY.register
class CPO(TRPO):
    """
    Paper Name: Constrained Policy Optimization Algorithm.
    Paper author: Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel
    Paper URL: https://arxiv.org/abs/1705.10528

    This implementation does not use cost shaping, but relies on exploration noise annealing.

    :param algo: Name of the algorithm
    :param cost_limit: Upper bound of max-cost, set as restriction
    :param use_cost_value_function: Whether to use two critics value and cost net and update them
    :param use_kl_early_stopping: Whether stop searching when kl-distance between target-policy and policy becomes too far
    :param loss_pi_cost_before: Pi and cost loss last iter
    """

    def __init__(self, algo='cpo', **cfgs):
        TRPO.__init__(self, algo=algo, **cfgs)
        cfgs = cfgs['cfgs']
        self.cost_limit = cfgs['cost_limit']
        self.loss_pi_cost_before = 0.0

    def search_step_size(
        self, step_dir, g_flat, c, optim_case, p_dist, data, total_steps=25, decay=0.8
    ):
        """
        CPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs.
        :param step_dir:direction theta changes towards
        :param g_flat:  gradient tensor of reward ,informs about how rewards improve with change of step direction
        :param c:how much epcost goes above limit
        :param p_dist: inform about old policy, how the old policy p performs on observation this moment
        :param optim_case: the way to optimize
        :param data: data buffer,mainly with adv, costs, values, actions, and observations
        :param decay: how search-step reduces in line-search
        """
        # Get distance each time theta goes towards certain direction
        step_frac = 1.0
        # Get and flatten parameters from pi-net
        _theta_old = get_flat_params_from(self.ac.pi.net)
        _, old_log_p = self.ac.pi(data['obs'], data['act'])
        # Reward improve expection,g-flat as gradient of reward
        expected_rew_improve = g_flat.dot(step_dir)

        # While not within_trust_region and not finish all steps:
        for j in range(total_steps):
            # New θ=θ_0+Δ Δ=Δdistance * direction
            new_theta = _theta_old + step_frac * step_dir
            # Set new θ as new pi-net's parameters
            set_param_values_to_model(self.ac.pi.net, new_theta)
            # The last acceptance steps to next step
            acceptance_step = j + 1

            with torch.no_grad():
                # Loss of policy reward from target/expected reward
                loss_pi_rew, _ = self.compute_loss_pi(data=data)
                # Loss of cost of policy cost from real/expected reward
                loss_pi_cost, _ = self.compute_loss_cost_performance(data=data)
                # Compute KL distance between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                # Kl_divergence with a form of pi*log(pi/qi)
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            # Δloss(rew)(p-q)
            loss_rew_improve = self.loss_pi_before - loss_pi_rew.item()
            # Δcost(p-q)
            cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            # Average across MPI processes...
            torch_kl = distributed_tools.mpi_avg(torch_kl)
            # Pi_average of torch_kl above
            loss_rew_improve = distributed_tools.mpi_avg(loss_rew_improve)
            cost_diff = distributed_tools.mpi_avg(cost_diff)

            self.logger.log(
                'Expected Improvement: %.3f Actual: %.3f' % (expected_rew_improve, loss_rew_improve)
            )
            # Check whether there are nans
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
            # If didnt find a step satisfys those conditions
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)
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
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        cost_loss = (ratio * data['cost_adv']).mean()
        # ent = dist.entropy().mean().item()
        info = {}
        return cost_loss, info

    def update_policy_net(self, data):
        # Get loss and info values before update (one-dimensional parameters of net=pi, as θ-old)
        theta_old = get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        # Get previous loss of value
        self.loss_v_before = self.compute_loss_v(data['obs'], data['target_v']).item()
        # Get previous loss of cost
        self.loss_c_before = self.compute_loss_c(data['obs'], data['target_c']).item()
        # Get prob. distribution before updates, previous dist of possibilities
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # Average grads across MPI processes
        distributed_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = get_flat_gradients_from(self.ac.pi.net)

        # Flip sign since policy_loss = -(ration * adv)
        g_flat *= -1
        # x: g or g_T in original paper, stands for gradient of cost funtion
        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))
        assert xHx.item() >= 0, 'No negative values'

        # get the policy cost performance gradient b (flat as vector)
        self.pi_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data=data)
        loss_cost.backward()
        # average grads across MPI processes
        distributed_tools.mpi_avg_grads(self.ac.pi.net)
        self.loss_pi_cost_before = loss_cost.item()
        b_flat = get_flat_gradients_from(self.ac.pi.net)
        # :param ep_costs: do samplings to get approximate costs as ep_costs
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # :params c: how much sampled result of cost goes beyond limit
        c = ep_costs - self.cost_limit
        # Rescale, and add eps(small float) to avoid nan
        c /= self.logger.get_stats('Metrics/EpLen')[0] + eps  # rescale

        # Set variable names as used in the paper with conjugate_gradient method,
        # used to solve equation(compute Hassen Matrix) instead of Natural Gradient
        p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx  # conjugate of matrix H
        r = g_flat.dot(p)  # g^T H^{-1} b
        s = b_flat.dot(p)  # b^T H^{-1} b

        # optim_case: divided into 5 kinds to compute
        if b_flat.dot(b_flat) <= 1e-6 and c < 0:
            # feasible step and cost grad is zero: use plain TRPO update...
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all()
            assert torch.isfinite(s).all()

            # A,b: mathematical value, not too much true meaning
            A = q - r**2 / s  # must be always >= 0 (Cauchy-Schwarz inequality)
            B = 2 * self.target_kl - c**2 / s  # safety line intersects trust-region if B > 0

            if c < 0 and B < 0:
                # point in trust region is feasible and safety boundary doesn't intersect
                # ==> entire trust region is feasible
                optim_case = 3
            elif c < 0 and B >= 0:
                # x = 0 is feasible and safety boundary intersects
                # ==> most of trust region is feasible
                optim_case = 2
            elif c >= 0 and B >= 0:
                # x = 0 is infeasible and safety boundary intersects
                # ==> part of trust region is feasible, recovery possible
                optim_case = 1
                self.logger.log('Alert! Attempting feasible recovery!', 'yellow')
            else:
                # x = 0 infeasible, and safety halfspace is outside trust region
                # ==> whole trust region is infeasible, try to fail gracefully
                optim_case = 0
                self.logger.log('Alert! Attempting infeasible recovery!', 'red')

        # the following computes required nu_star and lambda_star
        if optim_case in [3, 4]:
            # under 3 and 4 cases directly use trpo method
            alpha = torch.sqrt(
                2 * self.target_kl / (xHx + 1e-8)
            )  # step gap fixed by KKT condition in conjugate algorithm
            nu_star = torch.zeros(1)
            lambda_star = 1 / alpha
            step_dir = alpha * x  # change step direction to gap * gradient

        elif optim_case in [1, 2]:
            # in 1 and 2,
            def project_on_set(t: torch.Tensor, low: float, high: float) -> torch.Tensor:
                return torch.Tensor([max(low, min(t, high))])

            #  Analytical Solution to LQCLP, employ lambda,nu to compute final solution of argmax-OLOLQC
            #  λ=argmax(f_a(λ),f_b(λ)) = λa_star or λb_star
            #  (computing formula shown in appendix) λa and λb
            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * self.target_kl))
            # λa_star = Proj(lambda_a ,0 ~ r/c)  λb_star=Proj(lambda_b,r/c~ +inf)
            # where Proj(str,b,c)=max(b,min(str,c)) , may be regarded as a projection from effective region towards safety region
            if c < 0:
                lambda_a_star = project_on_set(lambda_a, 0.0, r / c)
                lambda_b_star = project_on_set(lambda_b, r / c, np.inf)
            else:
                lambda_a_star = project_on_set(lambda_a, r / c, np.inf)
                lambda_b_star = project_on_set(lambda_b, 0.0, r / c)

            def f_a(lam):
                return -0.5 * (A / (lam + eps) + B * lam) - r * c / (s + eps)

            def f_b(lam):
                return -0.5 * (q / (lam + eps) + 2 * self.target_kl * lam)

            lambda_star = (
                lambda_a_star if f_a(lambda_a_star) >= f_b(lambda_b_star) else lambda_b_star
            )

            # Discard all negative values with torch.clamp(x, min=0)
            # Nu_star = (lambda_star * - r)/s
            nu_star = torch.clamp(lambda_star * c - r, min=0) / (s + eps)
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
            c=c,
            optim_case=optim_case,
            p_dist=p_dist,
            data=data,
            total_steps=20,
        )
        # Update actor network parameters
        # just like trpo update method
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.ac.pi.net, new_theta)
        # Output the performance of pi policy on observation
        q_dist = self.ac.pi.dist(data['obs'])
        torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()

        self.logger.store(
            **{
                'Values/Adv': data['act'].numpy(),
                'Entropy': pi_info['ent'],
                'KL': torch_kl,
                'PolicyRatio': pi_info['ratio'],
                'Loss/Pi': self.loss_pi_before,
                'Loss/DeltaPi': loss_pi.item() - self.loss_pi_before,
                'Loss/Cost': 0.0,
                'Loss/DeltaCost': 0.0,
                'Misc/StopIter': 1,
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
