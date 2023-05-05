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
"""Implementation of CRR."""

from typing import Any, Callable, Dict, Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.dataset import OfflineDatasetWithInit
from omnisafe.models.actor.actor_builder import ActorBuilder
from omnisafe.models.offline import ObsEncoder


@registry.register
class COptiDICE(BaseOffline):  # pylint: disable=too-many-instance-attributes
    """COptiDICE: Offline Constrained Reinforcement Learning via Stationary Distribution Correction Estimation.

    References:
        - Title: COptiDICE: Offline Constrained Reinforcement Learning via Stationary
                Distribution Correction Estimation
        - Author: Lee, JongminPaduraru, CosminMankowitz, Daniel JHeess, NicolasPrecup, Doina
        - URL: `https://arxiv.org/abs/2204.08957`
    """

    def _init(self) -> None:
        self._dataset = OfflineDatasetWithInit(
            self._cfgs.train_cfgs.dataset,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            device=self._device,
        )
        self._fn, self._fn_inv = self._get_f_divergence_fn(self._cfgs.algo_cfgs.fn_type)

    def _init_log(self) -> None:
        """Log the COptiDICE specific information.

        +----------------------------+--------------------------------------------+
        | Things to log              | Description                                |
        +============================+============================================+
        | Loss/Loss_actor            | Loss of the actor network.                 |
        +----------------------------+--------------------------------------------+
        | Loss/Loss_Nu               | Loss of the Nu network, used in CoptiDICE. |
        +----------------------------+--------------------------------------------+
        | Loss/Loss_chi              | Loss of the chi network, used in COptiDICE.|
        +----------------------------+--------------------------------------------+
        | Loss/Loss_lamb             | Loss of the lambda multiplier.             |
        +----------------------------+--------------------------------------------+
        | Loss/Loss_Tau              | Loss of the Tau network, used in COptiDICE.|
        +----------------------------+--------------------------------------------+
        | Train/CostUB               | Cost up-bound                              |
        +----------------------------+--------------------------------------------+
        | Train/KL_divergence        | kl_divergence used in CotpiDICE.           |
        +----------------------------+--------------------------------------------+
        | Train/tau                  | :math:`tau` used in COptiDICE.             |
        +----------------------------+--------------------------------------------+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier.                   |
        +----------------------------+--------------------------------------------+
        | Metrics/PolicyStd          | The standard deviation of the policy.      |
        +----------------------------+--------------------------------------------+
        """
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'actor': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)

        self._logger.register_key('Loss/Loss_actor')
        self._logger.register_key('Loss/Loss_Nu')
        self._logger.register_key('Loss/Loss_Chi')
        self._logger.register_key('Loss/Loss_Lamb')
        self._logger.register_key('Loss/Loss_Tau')

        self._logger.register_key('Train/CostUB')
        self._logger.register_key('Train/KL_divergence')
        self._logger.register_key('Train/tau')
        self._logger.register_key('Train/lagrange_multiplier')
        self._logger.register_key('Train/PolicyStd')

    def _init_model(self) -> None:
        self._actor = (
            ActorBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.actor.hidden_sizes,
                activation=self._cfgs.model_cfgs.actor.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
            )
            .build_actor('gaussian_learning')
            .to(self._device)
        )
        assert isinstance(
            self._cfgs.model_cfgs.actor.lr,
            float,
        ), 'The learning rate must be a float number.'
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._cfgs.model_cfgs.actor.lr,
        )

        self._nu_net = ObsEncoder(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.nu.hidden_sizes,
            activation=self._cfgs.model_cfgs.nu.activation,
            weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
            out_dim=1,
        ).to(self._device)
        self._nu_net_optimizer = optim.Adam(
            self._nu_net.parameters(),
            lr=self._cfgs.model_cfgs.nu.lr,
        )

        self._chi_net = ObsEncoder(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            hidden_sizes=self._cfgs.model_cfgs.chi.hidden_sizes,
            activation=self._cfgs.model_cfgs.chi.activation,
            weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
            out_dim=1,
        ).to(self._device)
        self._chi_net_optimizer = optim.Adam(
            self._chi_net.parameters(),
            lr=self._cfgs.model_cfgs.chi.lr,
        )

        lamb_init = torch.as_tensor(
            self._cfgs.model_cfgs.lamb.init,
            dtype=torch.float32,
            device=self._device,
        )
        self._lamb = nn.Parameter(torch.clamp(lamb_init, 0, 1e3), requires_grad=True)
        self._lamb_optimizer = optim.Adam(
            params=[self._lamb],
            lr=self._cfgs.model_cfgs.lamb.lr,
        )

        tau_init = torch.as_tensor(
            self._cfgs.model_cfgs.tau.init,
            dtype=torch.float32,
            device=self._device,
        )
        self._tau = nn.Parameter(tau_init + 1e-6, requires_grad=True)
        self._tau_optimizer = optim.Adam(
            params=[self._tau],
            lr=self._cfgs.model_cfgs.tau.lr,
        )

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, action, reward, cost, next_obs, done, init_obs = batch

        # train nu, chi, lamb, tau
        self._update_net(obs, reward, cost, done, next_obs, init_obs)

        # train actor
        self._update_actor(obs, action, reward, cost, done, next_obs)

    # pylint: disable=too-many-locals
    def _update_net(
        self,
        obs: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
        init_obs: torch.Tensor,
    ) -> None:
        batch_size = obs.shape[0]

        nu = self._nu_net.forward(obs)
        nu_next = self._nu_net.forward(next_obs)
        adv = self._advantage(reward, cost, done, nu, nu_next)
        w_sa = self._w_sa(adv)

        nu_loss = (
            (1 - self._cfgs.algo_cfgs.gamma) * self._nu_net.forward(init_obs).mean()
            - self._cfgs.algo_cfgs.alpha * self._fn(w_sa).mean()
            + (w_sa * adv).mean()
        )

        chi = self._chi_net.forward(obs)  # (batch_size, )
        chi_next = self._chi_net.forward(next_obs)  # (batch_size, )
        chi_init = self._chi_net.forward(init_obs)  # (batch_size, )
        w_sa_no_grad = w_sa.detach()

        ell = (1 - self._cfgs.algo_cfgs.gamma) * chi_init + w_sa_no_grad * (
            cost + self._cfgs.algo_cfgs.gamma * (1 - done) * chi_next - chi
        )
        logist = ell / self._tau.item()
        weights = nn.functional.softmax(logist, dim=0) * batch_size
        log_weights = nn.functional.log_softmax(logist, dim=0) + torch.log(
            torch.as_tensor(batch_size, device=self._device),
        )
        kl_divergence = (weights * log_weights - weights + 1).mean()
        cost_ub = (w_sa_no_grad * cost).mean()
        chi_loss = (weights * ell).mean()
        tau_loss = -self._tau * (kl_divergence.detach() - self._cfgs.algo_cfgs.cost_ub_eps)

        lamb_loss = -(self._lamb * (cost_ub.detach() - self._cfgs.algo_cfgs.cost_limit)).mean()

        self._nu_net_optimizer.zero_grad()
        nu_loss.backward()
        self._nu_net_optimizer.step()

        self._chi_net_optimizer.zero_grad()
        chi_loss.backward()
        self._chi_net_optimizer.step()

        self._lamb_optimizer.zero_grad()
        lamb_loss.backward()
        self._lamb_optimizer.step()
        self._lamb.data.clamp_(min=0, max=1e3)

        self._tau_optimizer.zero_grad()
        tau_loss.backward()
        self._tau_optimizer.step()
        self._tau.data.clamp_(min=1e-6)

        self.logger.store(
            **{
                'Loss/Loss_Nu': nu_loss.item(),
                'Loss/Loss_Chi': chi_loss.item(),
                'Loss/Loss_Lamb': lamb_loss.item(),
                'Loss/Loss_Tau': tau_loss.item(),
                'Train/CostUB': cost_ub.item(),
                'Train/KL_divergence': kl_divergence.item(),
                'Train/tau': self._tau.item(),
                'Train/lagrange_multiplier': self._lamb.item(),
            },
        )

    def _update_actor(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        reward: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        self._actor.predict(obs)
        logp = self._actor.log_prob(act)

        nu = self._nu_net.forward(obs)
        nu_next = self._nu_net.forward(next_obs)
        adv = self._advantage(reward, cost, done, nu, nu_next)
        w_sa = self._w_sa(adv)

        policy_loss = -(w_sa * logp).mean()
        self._actor_optimizer.zero_grad()
        policy_loss.backward()
        self._actor_optimizer.step()

        self.logger.store(
            {
                'Loss/Loss_actor': policy_loss.item(),
                'Train/PolicyStd': self._actor.std,
            },
        )

    def _advantage(
        self,
        rewrad: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        nu: torch.Tensor,
        nu_next: torch.Tensor,
    ) -> torch.Tensor:  # pylint: disable=too-many-arguments
        return (
            rewrad
            - self._lamb.item() * cost
            + (1 - done) * self._cfgs.algo_cfgs.gamma * nu_next
            - nu
        )

    def _w_sa(self, adv: torch.Tensor) -> torch.Tensor:
        return nn.functional.relu(self._fn_inv(adv / self._cfgs.algo_cfgs.alpha))

    @staticmethod
    def _get_f_divergence_fn(
        fn_type: str,
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        if fn_type == 'kl':

            def fn(x: torch.Tensor) -> torch.Tensor:
                return x * torch.log(x + 1e-10)

            def fn_inv(x: torch.Tensor) -> torch.Tensor:
                return torch.exp(x - 1)

        elif fn_type == 'softchi':

            def fn(x: torch.Tensor) -> torch.Tensor:
                return torch.where(x < 1, x * (torch.log(x + 1e-10) - 1) + 1, 0.5 * (x - 1) ** 2)

            def fn_inv(x: torch.Tensor) -> torch.Tensor:
                return torch.where(x < 0, torch.exp(torch.min(x, torch.zeros_like(x))), x + 1)

        elif fn_type == 'chisquare':

            def fn(x: torch.Tensor) -> torch.Tensor:
                return 0.5 * (x - 1) ** 2

            def fn_inv(x: torch.Tensor) -> torch.Tensor:
                return x + 1

        return fn, fn_inv
