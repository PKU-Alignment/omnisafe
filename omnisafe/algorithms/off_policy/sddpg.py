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
"""Implementation of the SDDPG algorithm."""

from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.utils.config import Config
from omnisafe.utils import distributed
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
# pylint: disable-next=too-many-instance-attributes
class SDDPG(DDPG):
    """Lyapunov-based Safe Policy Optimization for Continuous Control.

    References:
        - Title: Lyapunov-based Safe Policy Optimization for Continuous Control
        - Authors: Yinlam Chow, Ofir Nachum, Aleksandra Faust, Edgar Duenez-Guzman,
                   Mohammad Ghavamzadeh.
        - URL: `SDDPG <https://arxiv.org/abs/1901.10031>`_
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        super().__init__(env_id, cfgs)

        self._fvp_obs: torch.Tensor

    def _fvp(self, params: torch.Tensor) -> torch.Tensor:
        """Build the `Hessian-vector product <https://en.wikipedia.org/wiki/Hessian_matrix>`_
        based on an approximation of the KL-divergence.

        The Hessian-vector product is approximated by the Fisher information matrix,
        which is the second-order derivative of the KL-divergence.

        For details see John Schulman's PhD thesis (pp. 40) http://joschu/docs/thesis.pdf

        Args:
            params (torch.Tensor): The parameters of the actor network.
        """
        self._actor_critic.actor.zero_grad()
        q_dist = self._actor_critic.actor(self._fvp_obs)
        with torch.no_grad():
            p_dist = self._actor_critic.actor(self._fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self._actor_critic.actor.parameters(), create_graph=True)  # type: ignore
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(kl_p, self._actor_critic.actor.parameters(), retain_graph=False)  # type: ignore

        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        distributed.avg_tensor(flat_grad_grad_kl)

        return flat_grad_grad_kl + params * self._cfgs.algo_cfgs.cg_damping

    def _loss_pi_cost(self, obs):
        r"""Compute the performance of cost on this moment.

        Detailedly, we compute the loss of cost of policy cost from real cost.

        .. math::
            L = \mathbb{E}_{\pi} \left[ \frac{\pi(a|s)}{\pi_{old}(a|s)} A^C(s, a) \right]

        where :math:`A^C(s, a)` is the cost advantage,
        :math:`\pi_{old}(a|s)` is the old policy,
        :math:`\pi(a|s)` is the current policy.

        Args:
            obs (torch.Tensor): Observation.
        """
        # compute loss
        action, _ = self._actor_critic.actor.predict(obs, deterministic=True)
        loss_pi = self._actor_critic.cost_critic(obs, action)[0]
        pi_info = {}
        return loss_pi.mean(), pi_info

    # pylint: disable-next=too-many-locals
    def _update_actor( 
        self,
        obs: torch.Tensor,
    ) -> None:  # pylint: disable=invalid-name
        """Update policy network.

        SDDPG updates policy network using the conjugate gradient algorithm,
        following the steps:

        - Compute the gradient of the policy.
        - Compute the step direction.
        - Search for a step size that satisfies the constraint. (Both KL divergence and cost limit).
        - Update the policy network.

        Args:
            obs (torch.Tensor): The observation tensor.
        """
        # Train policy with one steps of gradient descent
        self.fvp_obs = obs
        theta_old = get_flat_params_from(self._actor_critic.actor)

        self._actor_critic.actor_optimizer.zero_grad()
        loss_pi, _ = self._loss_pi(obs)
        loss_pi.backward()

        g_flat = get_flat_gradients_from(self._actor_critic.actor)
        g_flat *= -1

        g_inv_x = conjugate_gradients(self.Fvp, g_flat, self.cfgs.cg_iters)
        assert torch.isfinite(g_inv_x).all()

        eps = 1.0e-8
        hessian_x = torch.dot(g_inv_x, self.Fvp(g_inv_x))

        alpha = torch.sqrt(2 * self.cfgs.target_kl / (hessian_x + eps))

        self._actor_critic.actor_optimizer.zero_grad()
        loss_cost, _ = self._loss_pi_cost(obs)
        loss_cost.backward()

        self.loss_record.append(loss_pi=(loss_pi - loss_cost).mean().item())

        b_flat = get_flat_gradients_from(self._actor_critic.actor)
        b_inv_d = conjugate_gradients(self.Fvp, b_flat, self.cfgs.cg_iters)
        hessian_d = torch.dot(b_inv_d, self.Fvp(b_inv_d))
        hessian_s = torch.dot(b_inv_d, self.Fvp(b_inv_d))

        epsilon = (1 - self.cfgs.gamma) * (self.cfgs.d_init - loss_cost)
        lambda_star = (-self.cfgs.beta * epsilon - hessian_s) / (hessian_d + eps)

        final_step_dir = (
            -alpha / self.cfgs.beta * (self.Fvp(g_inv_x) - lambda_star * self.Fvp(b_inv_d))
        )
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self._actor_critic.actor, new_theta)
