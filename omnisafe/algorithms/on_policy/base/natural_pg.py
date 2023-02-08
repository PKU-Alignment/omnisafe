# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Natural Policy Gradient algorithm."""

from typing import NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.utils import distributed_utils
from omnisafe.utils.tools import (
    conjugate_gradients,
    get_flat_gradients_from,
    get_flat_params_from,
    set_param_values_to_model,
)


@registry.register
class NaturalPG(PolicyGradient):
    """The Natural Policy Gradient algorithm.

    The Natural Policy Gradient algorithm is a policy gradient algorithm that uses the
    `Fisher information matrix <https://en.wikipedia.org/wiki/Fisher_information>`_ to
    approximate the Hessian matrix. The Fisher information matrix is the second-order derivative of the KL-divergence.

    References:
        - Title: A Natural Policy Gradient
        - Author: Sham Kakade.
        - URL: `Natural PG <https://proceedings.neurips.cc/paper/2001/file/4b86abe48d358ecf194c56c69108433e-Paper.pdf>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize Natural Policy Gradient.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)
        self.cg_damping = cfgs.cg_damping
        self.cg_iters = cfgs.cg_iters
        self.target_kl = cfgs.target_kl
        self.fvp_obs = cfgs.fvp_obs

    def search_step_size(self, step_dir: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """NPG use full step_size, so we just return 1.

        Args:
            step_dir (torch.Tensor): The step direction.
        """
        accept_step = 1
        return step_dir, accept_step

    def algorithm_specific_logs(self) -> None:
        r"""Log the Natural Policy Gradient specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   ``Misc/AcceptanceStep``
                -   The acceptance step size.
            *   -   ``Misc/Alpha``
                -   :math:`\frac{\delta_{KL}}{xHx}` in original paper.
                    where :math:`x` is the step direction, :math:`H` is the Hessian matrix,
                    and :math:`\delta_{KL}` is the target KL divergence.
            *   -   ``Misc/FinalStepNorm``
                -   The final step norm.
            *   -   ``Misc/gradient_norm``
                -   The gradient norm.
            *   -   ``Misc/xHx``
                -   :math:`xHx` in original paper.
            *   -   ``Misc/H_inv_g``
                -   :math:`H^{-1}g` in original paper.

        """
        self.logger.log_tabular('Misc/AcceptanceStep')
        self.logger.log_tabular('Misc/Alpha')
        self.logger.log_tabular('Misc/FinalStepNorm')
        self.logger.log_tabular('Misc/gradient_norm')
        self.logger.log_tabular('Misc/xHx')
        self.logger.log_tabular('Misc/H_inv_g')

    def Fvp(self, params: torch.Tensor) -> torch.Tensor:
        """Build the `Hessian-vector product <https://en.wikipedia.org/wiki/Hessian_matrix>`_
        based on an approximation of the KL-divergence.
        The Hessian-vector product is approximated by the Fisher information matrix,
        which is the second-order derivative of the KL-divergence.
        For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf

        Args:
            params (torch.Tensor): The parameters of the actor network.
        """
        self.actor_critic.actor.zero_grad()
        q_dist = self.actor_critic.actor(self.fvp_obs)
        with torch.no_grad():
            p_dist = self.actor_critic.actor(self.fvp_obs)
        kl = torch.distributions.kl.kl_divergence(p_dist, q_dist).mean()

        grads = torch.autograd.grad(kl, self.actor_critic.actor.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        kl_p = (flat_grad_kl * params).sum()
        grads = torch.autograd.grad(kl_p, self.actor_critic.actor.parameters(), retain_graph=False)
        # contiguous indicating, if the memory is contiguously stored or not
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])
        distributed_utils.mpi_avg_torch_tensor(flat_grad_grad_kl)
        return flat_grad_grad_kl + params * self.cg_damping

    # pylint: disable-next=too-many-locals,too-many-arguments
    def update_policy_net(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
        cost_adv: torch.Tensor,
    ) -> None:
        """Update policy network.

        Natural Policy Gradient (NPG) update policy network using the conjugate gradient algorithm,
        following the steps:

        - Calculate the gradient of the policy network,
        - Use the conjugate gradient algorithm to calculate the step direction.
        - Use the line search algorithm to find the step size.

        Args:
            obs (torch.Tensor): The observation tensor.
            act (torch.Tensor): The action tensor.
            log_p (torch.Tensor): The log probability of the action.
            adv (torch.Tensor): The advantage tensor.
            cost_adv (torch.Tensor): The cost advantage tensor.
        """
        # get loss and info values before update
        self.fvp_obs = obs[::4]
        theta_old = get_flat_params_from(self.actor_critic.actor)
        self.actor_critic.actor.zero_grad()
        processed_adv = self.compute_surrogate(adv=adv, cost_adv=cost_adv)
        loss_pi, pi_info = self.compute_loss_pi(
            obs=obs,
            act=act,
            log_p=log_p,
            adv=processed_adv,
        )
        # train policy with multiple steps of gradient descent
        loss_pi.backward()
        # average grads across MPI processes
        distributed_utils.mpi_avg_grads(self.actor_critic.actor)
        g_flat = get_flat_gradients_from(self.actor_critic.actor)
        g_flat *= -1

        # pylint: disable-next=invalid-name
        x = conjugate_gradients(self.Fvp, g_flat, self.cg_iters)
        assert torch.isfinite(x).all()
        # note that xHx = g^T x, but calculating xHx is faster than g^T x
        xHx = torch.dot(x, self.Fvp(x))  # equivalent to : g^T x
        assert xHx.item() >= 0, 'No negative values'

        # perform descent direction
        alpha = torch.sqrt(2 * self.target_kl / (xHx + 1e-8))
        step_direction = alpha * x
        assert torch.isfinite(step_direction).all()

        # determine step direction and apply SGD step after grads where set
        # TRPO uses custom backtracking line search
        final_step_dir, accept_step = self.search_step_size(step_dir=step_direction)

        # update actor network parameters
        new_theta = theta_old + final_step_dir
        set_param_values_to_model(self.actor_critic.actor, new_theta)

        with torch.no_grad():
            loss_pi, pi_info = self.compute_loss_pi(obs=obs, act=act, log_p=log_p, adv=adv)
            self.loss_record.append(loss_pi=loss_pi.mean().item())

        self.logger.store(
            **{
                'Train/Entropy': pi_info['ent'],
                'Train/PolicyRatio': pi_info['ratio'],
                'Misc/AcceptanceStep': accept_step,
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(final_step_dir).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(g_flat).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )
