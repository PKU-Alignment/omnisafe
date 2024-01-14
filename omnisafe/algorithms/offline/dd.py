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
"""Implementation of BCQ."""

from copy import deepcopy
from typing import Any, Dict, Tuple

import torch
from torch import nn, optim

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.dataset import DeciDiffuserDataset
from omnisafe.models.dd_models.diffusion import GaussianInvDynDiffusion
from omnisafe.models.dd_models.temporal import TemporalUnet
from omnisafe.common.offline.dataset import OfflineDataset


@registry.register
class DD(BaseOffline):
    """Batch-Constrained Deep Reinforcement Learning.

    References:
        - Title: Off-Policy Deep Reinforcement Learning without Exploration
        - Author: Fujimoto, ScottMeger, DavidPrecup, Doina.
        - URL: `https://arxiv.org/abs/1812.02900`
    """

    def _init_log(self) -> None:
        """Log the BCQ specific information.

        +-------------------------+----------------------------------------------------+
        | Things to log           | Description                                        |
        +=========================+====================================================+
        | Loss/Loss_vae           | Loss of VAE network                                |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_recon         | Reconstruction loss of VAE network                 |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_kl            | KL loss of VAE network                             |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_actor         | Loss of the actor network.                         |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_reward_critic | Loss of the reward critic.                         |
        +-------------------------+----------------------------------------------------+
        | Qr/data_Qr              | Average Q value of offline data.                   |
        +-------------------------+----------------------------------------------------+
        | Qr/target_Qr            | Average Q value of next_obs and next_action.       |
        +-------------------------+----------------------------------------------------+
        | Qr/current_Qr           | Average Q value of obs and agent predicted action. |
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

    def _init(self,
              ema_decay=0.995,
              train_batch_size=32,
              train_lr=2e-5,
              gradient_accumulate_every=2,
              log_freq=100,
              sample_freq=1000,
              save_freq=1000,
              label_freq=100000,
              save_parallel=False,
              n_reference=8,
              bucket=None,
              train_device='cuda',
              save_checkpoints=False,
              ) -> None:
        self.update_ema_every = 2000
        self.save_checkpoints = self._cfgs.save_checkpoints

        self.step_start_ema = 10
        self.log_freq = self._cfgs.log_freq
        self.sample_freq = self._cfgs.sample_freq
        self.save_freq = self._cfgs.save_freq
        self.label_freq = int(self._cfgs.n_train_steps // self._cfgs.n_saves)
        self.save_parallel = self._cfgs.save_parallel

        self.batch_size = self._cfgs.batch_size
        self.gradient_accumulate_every = self._cfgs.gradient_accumulate_every

        self.bucket = self._cfgs.bucket
        self.n_reference = self._cfgs.n_reference

        self.step = 0

        self.device = train_device
        self._dataset = DeciDiffuserDataset(self._cfgs.train_cfgs.dataset,
                                            batch_size=self._cfgs.algo_cfgs.batch_size,
                                            device=self._device,
                                            horizon=self._cfgs.horizon,
                                            discount=self._cfgs.discount,
                                            returns_scale=self._cfgs.returns_scale,
                                            include_returns=self._cfgs.include_returns,
                                            )

    def _init_model(self) -> None:

        observation_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]
        TUmodel = TemporalUnet(
            horizon=self._cfgs.horizon,
            transition_dim=observation_dim,
            cond_dim=observation_dim,
            dim_mults=self._cfgs.dim_mults,
            returns_condition=self._cfgs.returns_condition,
            dim=self._cfgs.dim,
            condition_dropout=self._cfgs.condition_dropout,
            calc_energy=self._cfgs.calc_energy,

        ).to(self._device)
        GDDModel = GaussianInvDynDiffusion(
            TUmodel,
            horizon=self._cfgs.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=self._cfgs.n_diffusion_steps,
            loss_type=self._cfgs.loss_type,
            clip_denoised=self._cfgs.clip_denoised,
            predict_epsilon=self._cfgs.predict_epsilon,
            action_weight=self._cfgs.action_weight,
            loss_weights=self._cfgs.loss_weights,
            loss_discount=self._cfgs.loss_discount,
            returns_condition=self._cfgs.returns_condition,
            condition_guidance_w=self._cfgs.condition_guidance_w,
        ).to(self._device)
        self._actor = GDDModel
        self._optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._cfgs.learning_rate)

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:

        for i in range(self.gradient_accumulate_every):
            loss, infos = self._actor.loss(*batch)
            loss = loss / self.gradient_accumulate_every
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

    def trans_dataset(self) -> dict:

        name_trans_dict = {
            'obs': 'observations',
            'next_obs': 'next_observations',
            'action': 'actions',
            'reward': 'rewards',
            'done': 'terminals',
            'cost': 'cost'
        }
        raw_dataset = OfflineDataset(
            self._cfgs.train_cfgs.dataset,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            device=self._device,
        )
        processed_dataset = {}
        for key, value in raw_dataset.__dict__.items():
            if not key[0] == '_':
                processed_dataset[name_trans_dict[key]] = value.cpu().numpy()
        return processed_dataset
