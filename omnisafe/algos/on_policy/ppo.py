import torch

from omnisafe.algos.on_policy.policy_gradient import PolicyGradient
from omnisafe.algos.registry import REGISTRY


@REGISTRY.register
class PPO(PolicyGradient):
    """
    Paper Name: Proximal Policy Optimization Algorithms
    Paper author: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
    Paper URL: https://arxiv.org/pdf/1707.06347.pdf
    """

    def __init__(self, algo='ppo', clip=0.2, **cfgs):
        PolicyGradient.__init__(self, algo=algo, **cfgs)
        self.clip = clip

    def compute_loss_pi(self, data: dict):
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        # Importance ratio
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2 / dist.stddev**2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

        return loss_pi, pi_info
