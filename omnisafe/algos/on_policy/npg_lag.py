import torch

from omnisafe.algos.common.lagrange import Lagrange
from omnisafe.algos.on_policy.natural_pg import NaturalPG
from omnisafe.algos.registry import REGISTRY


@REGISTRY.register
class NPGLag(NaturalPG, Lagrange):
    def __init__(self, algo: str = 'pdo_ngp', **cfgs):

        NaturalPG.__init__(self, algo=algo, **cfgs)
        Lagrange.__init__(self, **self.cfgs['lagrange_cfgs'])

    def compute_loss_pi(self, data: dict):
        """
        computing pi/actor loss

        Returns:
            torch.Tensor
        """
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        # Compute loss via ratio and advantage
        penalty_lambda = self.lambda_range_projection(self.lagrangian_multiplier).item()
        adv = data['adv'] - penalty_lambda * data['cost_adv']
        loss_pi = -(ratio * adv).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        """
        Update actor, critic, running statistics
        """
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('Metrics/EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # Update Policy Network
        self.update_policy_net(data)
        # Update Value Function
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier', self.lagrangian_multiplier.item())
