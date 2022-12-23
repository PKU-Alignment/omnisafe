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
"""Implementation of the SDDPG algorithm."""

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class SDDPG(DDPG):  # pylint: disable=too-many-instance-attributes,invalid-name
    """Implementation of the SDDPG algorithm.

    References:
        Title: Lyapunov-based Safe Policy Optimization for Continuous Control
        Authors: Yinlam Chow, Ofir Nachum, Aleksandra Faust, Edgar Duenez-Guzman,
                 Mohammad Ghavamzadeh.
        URL: https://arxiv.org/abs/1901.10031
    """

    def __init__(self, env_id: str, cfgs=None) -> None:
        """Initialize SDDPG.

        Args:
            env_id (str): environment id.
            cfgs (dict): configurations.
            algo (str): algorithm name.
            wrapper_type (str): environment wrapper type.
        """
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )
        self.beta = cfgs.beta
        self.cg_damping = cfgs.cg_damping
        self.cg_iters = cfgs.cg_iters
        self.fvp_obs = None
        self.target_kl = cfgs.target_kl
        self.gamma = cfgs.gamma
        self.d_init = cfgs.d_init

    def update(self, data):
        """Update.

        Args:
            data (dict): data dictionary.
        """
        # First run one gradient descent step for Q.
        self.fvp_obs = data['obs'][::4]
        self.update_value_net(data)
        if self.cfgs.use_cost:
            self.update_cost_net(data)
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = False

        # Freeze Q-network so you don't waste computational effort
        # computing gradients for it during the policy learning step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = False

        # Next run one gradient descent step for actor.
        self.update_policy_net(data)

        # Unfreeze Q-network so you can optimize it at next DDPG step.
        for param in self.actor_critic.critic.parameters():
            param.requires_grad = True

        if self.cfgs.use_cost:
            for param in self.actor_critic.cost_critic.parameters():
                param.requires_grad = True

        # Finally, update target networks by polyak averaging.
        self.polyak_update_target()

    def Fvp(self, params):
        """
        Build the Hessian-vector product based on an approximation of the KL-divergence.
        For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf

        Args:
            params (torch.Tensor): parameters.

        Returns:
            flat_grad_grad_kl (torch.Tensor): flat gradient of gradient of KL.
        """
        self.actor_critic.actor.net.zero_grad()
        q_dist = self.actor_critic.actor.get_distribution(self.fvp_obs)
        with torch.no_grad():
            p_dist = self.actor_critic.actor.get_distribution(self.fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.actor_critic.actor.net.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(
            kl_p, self.actor_critic.actor.net.parameters(), retain_graph=False
        )
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        distributed_utils.mpi_avg_torch_tensor(flat_grad_grad_kl)
        return flat_grad_grad_kl + params * self.cg_damping

    def compute_loss_cost_performance(self, data):
        """Compute loss of cost performance.

        Args:
            data (dict): data dictionary.

        Returns:
            loss (torch.Tensor): loss of cost performance.
        """
        # Compute loss
        action, _ = self.actor_critic.actor.predict(data['obs'], deterministic=True)
        loss_pi = self.actor_critic.cost_critic(data['obs'], action)[0]
        pi_info = {}
        return loss_pi.mean(), pi_info

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def update_policy_net(self, data) -> None:
        """Update policy network.

        Args:
            data (dict): data dictionary.
        """
        # Train policy with one steps of gradient descent
        theta_old = get_flat_params_from(self.actor_critic.actor.net)

        self.actor_optimizer.zero_grad()
        loss_pi, _ = self.compute_loss_pi(data)
        loss_pi.backward()

        g_flat = get_flat_gradients_from(self.actor_critic.actor.net)
        g_flat *= -1

        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()

        eps = 1.0e-8
        xHx = torch.dot(x, self.Fvp(x))

        alpha = torch.sqrt(2 * self.target_kl / (xHx + eps))

        self.actor_optimizer.zero_grad()
        loss_cost, _ = self.compute_loss_cost_performance(data)
        loss_cost.backward()

        b_flat = get_flat_gradients_from(self.actor_critic.actor.net)
        d = conjugate_gradients(self.Fvp, b_flat, self.cg_iters)
        dHd = torch.dot(d, self.Fvp(d))
        sHd = torch.dot(d, self.Fvp(d))

        epsilon = (1 - self.gamma) * (self.d_init - loss_cost)
        lambda_star = (-self.beta * epsilon - sHd) / (dHd + eps)

        final_step_dir = -alpha / self.beta * (self.Fvp(x) - lambda_star * self.Fvp(d))
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor.net, new_theta)

        self.logger.store(**{'Loss/Pi': loss_pi.item()})
