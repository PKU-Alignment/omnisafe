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
"""Implementation of VAE Behavior Cloning."""

from typing import Any, Dict, Tuple

import torch
from torch import optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.models.actor import VAE
from omnisafe.models.actor.actor_builder import ActorBuilder


@registry.register
class VAEBC(BaseOffline):
    """Behavior Cloning with Variational Autoencoder.

    References:
        - Title: Off-Policy Deep Reinforcement Learning without Exploration
        - Author: Fujimoto, ScottMeger, DavidPrecup, Doina.
        - URL: `https://arxiv.org/abs/1812.02900`
    """

    def _init_log(self) -> None:
        """Log the VAE-BC specific information.

        +-------------------------+----------------------------------------------------+
        | Things to log           | Description                                        |
        +=========================+====================================================+
        | Loss/Loss_vae           | Loss of VAE network                                |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_recon         | Reconstruction loss of VAE network                 |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_kl            | KL loss of VAE network                             |
        +-------------------------+----------------------------------------------------+
        """
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'vae': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)

        self._logger.register_key('Loss/Loss_vae')
        self._logger.register_key('Loss/Loss_recon')
        self._logger.register_key('Loss/Loss_kl')

    def _init_model(self) -> None:
        self._actor: VAE = (
            ActorBuilder(  # type: ignore
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=self._cfgs.model_cfgs.hidden_sizes,
                activation=self._cfgs.model_cfgs.activation,
                weight_initialization_mode=self._cfgs.model_cfgs.weight_initialization_mode,
            )
            .build_actor(actor_type='vae')
            .to(self._device)
        )

        self._vae_optimizer = optim.Adam(
            self._actor.parameters(),
            lr=self._cfgs.model_cfgs.learning_rate,
        )

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        obs, act, _, _, _, _ = batch

        recon_loss, kl_loss = self._actor.loss(obs, act)
        loss = recon_loss + kl_loss
        self._vae_optimizer.zero_grad()
        loss.backward()
        self._vae_optimizer.step()

        self._logger.store(
            **{
                'Loss/Loss_vae': loss.item(),
                'Loss/Loss_recon': recon_loss.item(),
                'Loss/Loss_kl': kl_loss.item(),
            },
        )
