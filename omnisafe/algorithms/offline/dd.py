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
"""Implementation of Decision Diffusion."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import nn

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.dataset import DeciDiffuserDataset
from omnisafe.models.dd_models.diffusion import GaussianInvDynDiffusion
from omnisafe.models.dd_models.temporal import TemporalUnet
from omnisafe.utils.model import initialize_layer


@registry.register
class DD(BaseOffline):
    """Decision Diffusion.

    References:
        - Title: Is Conditional Generative Modeling all you need for Decision-Making?
        - Author: Ajay, Anurag and Du, Yilun and Gupta, Abhi and Tenenbaum, Joshua and Jaakkola, Tommi and Agrawal, Pulkit.
        - URL: `https://arxiv.org/abs/2211.15657`
    """

    def _init_log(self) -> None:
        """Log the DD specific information.

        +-------------------------+----------------------------------------------------+
        | Things to log           | Description                                        |
        +=========================+====================================================+
        | Loss/Loss_diffuser      | Loss of diffuser model                             |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_inv           | Loss of action inverse dynamic model               |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_total         | Avenger of Loss_diffuser and Loss_inv              |
        +-------------------------+----------------------------------------------------+

        """
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'deci_diffuser': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)
        self._logger.register_key('Loss/Loss_diffuser')
        self._logger.register_key('Loss/Loss_inv')
        self._logger.register_key('Loss/Loss_total')

    def _init(self) -> None:
        self._dataset = DeciDiffuserDataset(self._cfgs.train_cfgs.dataset,
                                            batch_size=self._cfgs.algo_cfgs.batch_size,
                                            device=self._device,
                                            horizon=self._cfgs.algo_cfgs.horizon,
                                            discount=self._cfgs.algo_cfgs.gamma,
                                            returns_scale=self._cfgs.dataset_cfgs.returns_scale,
                                            include_returns=self._cfgs.dataset_cfgs.include_returns,
                                            )

    def _init_model(self) -> None:

        observation_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]
        TUmodel = TemporalUnet(
            horizon=self._cfgs.algo_cfgs.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=self._cfgs.model_cfgs.temporalU_model.dim_mults,
            returns_condition=self._cfgs.model_cfgs.returns_condition,
            dim=self._cfgs.model_cfgs.temporalU_model.dim,
            condition_dropout=self._cfgs.model_cfgs.temporalU_model.condition_dropout,
            calc_energy=self._cfgs.model_cfgs.temporalU_model.calc_energy,

        ).to(self._device)
        GDDModel = GaussianInvDynDiffusion(
            TUmodel,
            horizon=self._cfgs.algo_cfgs.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=self._cfgs.algo_cfgs.n_diffusion_steps,
            loss_type=self._cfgs.train_cfgs.loss_type,
            clip_denoised=self._cfgs.model_cfgs.diffuser_model.clip_denoised,
            predict_epsilon=self._cfgs.model_cfgs.diffuser_model.predict_epsilon,
            action_weight=self._cfgs.model_cfgs.diffuser_model.action_weight,
            loss_weights=self._cfgs.model_cfgs.diffuser_model.loss_weights,
            hidden_dim=self._cfgs.model_cfgs.diffuser_model.hidden_dim,
            loss_discount=self._cfgs.model_cfgs.diffuser_model.loss_discount,
            returns_condition=self._cfgs.model_cfgs.returns_condition,
            ar_inv=self._cfgs.model_cfgs.diffuser_model.ar_inv,
            train_only_inv=self._cfgs.model_cfgs.diffuser_model.train_only_inv,
            condition_guidance_w=self._cfgs.model_cfgs.diffuser_model.condition_guidance_w,
            test_ret=self._cfgs.model_cfgs.diffuser_model.test_ret
        ).to(self._device)
        for name, layer in GDDModel.named_modules():
            if isinstance(layer, nn.Linear):
                initialize_layer(self._cfgs.model_cfgs.weight_initialization_mode, layer)

        self._actor = GDDModel
        self._optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._cfgs.model_cfgs.lr)

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:

        for i in range(self._cfgs.train_cfgs.gradient_accumulate_every):
            loss, infos = self._actor.loss(*batch)
            loss = loss / self._cfgs.train_cfgs.gradient_accumulate_every
            loss.backward()

        self._logger.store(
            **{
                'Loss/Loss_diffuser': infos["loss_diffuser"].item(),
                'Loss/Loss_inv': infos["loss_inv"].item(),
                'Loss/Loss_total': infos["loss_total"].item(),
            },
        )
        self._optimizer.step()
        self._optimizer.zero_grad()
