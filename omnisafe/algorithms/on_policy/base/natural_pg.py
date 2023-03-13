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
"""Implementation of the Natural Policy Gradient algorithm."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from omnisafe.utils.math import conjugate_gradients
from omnisafe.utils.tools import (
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

    def __init__(self, env_id: str, cfgs: Config) -> None:
        super().__init__(env_id, cfgs)

        self._fvp_obs: torch.Tensor

    def _init_log(self) -> None:
        super()._init_log()

        self._logger.register_key('Misc/Alpha')
        self._logger.register_key('Misc/FinalStepNorm')
        self._logger.register_key('Misc/gradient_norm')
        self._logger.register_key('Misc/xHx')
        self._logger.register_key('Misc/H_inv_g')

    def _fvp(self, params: torch.Tensor) -> torch.Tensor:
        """Build the `Hessian-vector product <https://en.wikipedia.org/wiki/Hessian_matrix>`_
        based on an approximation of the KL-divergence.
        The Hessian-vector product is approximated by the Fisher information matrix,
        which is the second-order derivative of the KL-divergence.
        For details see John Schulman's PhD thesis (pp. 40) http://joschu.net/docs/thesis.pdf

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

        self._logger.store(
            **{
                'Train/KL': kl.item(),
            }
        )
        return flat_grad_grad_kl + params * self._cfgs.algo_cfgs.cg_damping

    def _update_actor(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_r: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> None:
        self._fvp_obs = obs[:: self._cfgs.algo_cfgs.fvp_sample_freq]
        theta_old = get_flat_params_from(self._actor_critic.actor)
        self._actor_critic.actor.zero_grad()
        adv = self._compute_adv_surrogate(adv_r, adv_c)
        loss, info = self._loss_pi(obs, act, logp, adv)

        loss.backward()
        distributed.avg_grads(self._actor_critic.actor)

        grad = -get_flat_gradients_from(self._actor_critic.actor)
        x = conjugate_gradients(self._fvp, grad, self._cfgs.algo_cfgs.cg_iters)
        assert torch.isfinite(x).all(), 'x is not finite'
        xHx = torch.dot(x, self._fvp(x))
        assert xHx.item() >= 0, 'xHx is negative'
        alpha = torch.sqrt(2 * self._cfgs.algo_cfgs.target_kl / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), 'step_direction is not finite'

        theta_new = theta_old + step_direction
        set_param_values_to_model(self._actor_critic.actor, theta_new)

        with torch.no_grad():
            loss, info = self._loss_pi(obs, act, logp, adv)

        self._logger.store(
            **{
                'Train/Entropy': info['entropy'],
                'Train/PolicyRatio': info['ratio'],
                'Train/PolicyStd': info['std'],
                'Loss/Loss_pi': loss.mean().item(),
                'Misc/Alpha': alpha.item(),
                'Misc/FinalStepNorm': torch.norm(step_direction).mean().item(),
                'Misc/xHx': xHx.item(),
                'Misc/gradient_norm': torch.norm(grad).mean().item(),
                'Misc/H_inv_g': x.norm().item(),
            }
        )

    def _update(self) -> None:
        data = self._buf.get()
        obs, act, logp, target_value_r, target_value_c, adv_r, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['target_value_c'],
            data['adv_r'],
            data['adv_c'],
        )
        self._update_actor(obs, act, logp, adv_r, adv_c)

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, target_value_r, target_value_c, adv_r, adv_c),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for i in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                act,
                logp,
                target_value_r,
                target_value_c,
                adv_r,
                adv_c,
            ) in dataloader:
                self._update_rewrad_critic(obs, target_value_r)
                if self._cfgs.algo_cfgs.use_cost:
                    self._update_cost_critic(obs, target_value_c)

        self._logger.store(
            **{
                'Train/StopIter': i + 1,
                'Value/Adv': adv_r.mean().item(),
                'Train/KL': 0.0,
            }
        )
