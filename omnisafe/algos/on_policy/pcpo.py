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
class PCPO(TRPO):
    """
    Paper name: Constrained Policy Optimization Algorithm.
    Paper author: Tsung-Yen Yang, Justinian Rosca, Karthik Narasimhan, Peter J. Ramadge
    Paper URL: https://arxiv.org/abs/2010.03152
    This implementation does not use cost shaping, but relies on exploration noise annealing.
    """

    def __init__(self, algo='pcpo', **cfgs):
        TRPO.__init__(self, algo=algo, **cfgs)
        self.cost_limit = self.cfgs['cost_limit']
        self.loss_pi_cost_before = 0.0

    def adjust_cpo_step_direction(
        self,
        step_dir,
        g_flat,
        c,
        optim_case,
        p_dist,
        data,
        total_steps: int = 25,
        decay: float = 0.8,
    ):
        """
        PCPO algorithm performs line-search to ensure constraint satisfaction for rewards and costs.
        """
        step_frac = 1.0
        _theta_old = get_flat_params_from(self.ac.pi.net)
        _, old_log_p = self.ac.pi(data['obs'], data['act'])
        expected_rew_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi_rew, _ = self.compute_loss_pi(data=data)
                loss_pi_cost, _ = self.compute_loss_cost_performance(data=data)
                # determine KL div between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean().item()
            loss_rew_improve = self.loss_pi_before - loss_pi_rew.item()
            cost_diff = loss_pi_cost.item() - self.loss_pi_cost_before

            # Average across MPI processes...
            torch_kl = distributed_tools.mpi_avg(torch_kl)
            loss_rew_improve = distributed_tools.mpi_avg(loss_rew_improve)
            cost_diff = distributed_tools.mpi_avg(cost_diff)

            self.logger.log(
                'Expected Improvement: %.3f Actual: %.3f' % (expected_rew_improve, loss_rew_improve)
            )

            if not torch.isfinite(loss_pi_rew) and not torch.isfinite(loss_pi_cost):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_rew_improve < 0 if optim_case > 1 else False:
                self.logger.log('INFO: did not improve improve <0')

            elif cost_diff > max(-c, 0):
                self.logger.log(f'INFO: no improve {cost_diff} > {max(-c, 0)}')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log(f'INFO: violated KL constraint {torch_kl} at step {j + 1}.')
            else:
                # step only if surrogate is improved and we are
                # within the trust region
                self.logger.log(f'Accept step at i={j + 1}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)
        return step_frac * step_dir, acceptance_step

    def algorithm_specific_logs(self):
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
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        cost_loss = (ratio * data['cost_adv']).mean()
        # ent = dist.entropy().mean().item()
        info = {}
        return cost_loss, info

    def update_policy_net(self, data):
        # Get loss and info values before update
        theta_old = get_flat_params_from(self.ac.pi.net)
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data=data)
        self.loss_pi_before = loss_pi.item()
        self.loss_v_before = self.compute_loss_v(data['obs'], data['target_v']).item()
        self.loss_c_before = self.compute_loss_c(data['obs'], data['target_c']).item()
        # get prob. distribution before updates
        p_dist = self.ac.pi.dist(data['obs'])
        # Train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_tools.mpi_avg_grads(self.ac.pi.net)
        g_flat = get_flat_gradients_from(self.ac.pi.net)

        # flip sign since policy_loss = -(ration * adv)
        g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        eps = 1.0e-8
        # Note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        H_inv_g = self.Fvp(x)
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

        ep_costs = self.logger.get_stats('Metrics/EpCosts')[0]
        c = ep_costs - self.cost_limit
        c /= self.logger.get_stats('Metrics/EpLen')[0] + eps  # rescale
        self.logger.log(f'c = {c}')
        self.logger.log(f'b^T b = {b_flat.dot(b_flat).item()}')

        # set variable names as used in the paper
        p = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        q = xHx
        r = g_flat.dot(p)  # g^T H^{-1} b
        s = b_flat.dot(p)  # b^T H^{-1} b
        """
        q = torch.matmul(H_inv_g, approx_g)
        print("q", q)

        c = self.logger.get_stats('EpCost')[0] - self.cost_lim

        H_inv_a = self.cg_solver(Hx, a)
        approx_a = torch.tensor(Hx(H_inv_a))
        s = torch.matmul(approx_a, H_inv_a)
        x = torch.sqrt(2 * self.max_kl / (q+EPS)) * H_inv_g - torch.clamp_min((torch.sqrt(2 * self.max_kl/q) * s + c) / s, torch.tensor(0.0)) * H_inv_a
        """
        # x = torch.sqrt(2 * self.max_kl / (q+EPS)) * H_inv_g - torch.clamp_min((torch.sqrt(2 * self.max_kl/q) * s + c) / s, torch.tensor(0.0)) * H_inv_a
        step_dir = (
            torch.sqrt(2 * self.target_kl / (q + 1e-8)) * H_inv_g
            - torch.clamp_min((torch.sqrt(2 * self.target_kl / q) * r + c) / s, torch.tensor(0.0))
            * p
        )

        final_step_dir, accept_step = self.adjust_cpo_step_direction(
            step_dir,
            g_flat,
            c=c,
            optim_case=2,
            p_dist=p_dist,
            data=data,
            total_steps=20,
        )
        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.ac.pi.net, new_theta)

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
                'Misc/StopIter': 1,
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': final_step_dir.norm().numpy(),
                'Misc/xHx': xHx.numpy(),
                'Misc/H_inv_g': x.norm().item(),  # H^-1 g
                'Misc/gradient_norm': torch.norm(g_flat).numpy(),
                'Misc/cost_gradient_norm': torch.norm(b_flat).numpy(),
                'Misc/Lambda_star': 1.0,
                'Misc/Nu_star': 1.0,
                'Misc/OptimCase': int(1),
                'Misc/A': 1.0,
                'Misc/B': 1.0,
                'Misc/q': q.item(),
                'Misc/r': r.item(),
                'Misc/s': s.item(),
            }
        )
