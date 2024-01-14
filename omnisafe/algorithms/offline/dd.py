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
from omnisafe.common.offline.dd_datasets.sequence import SequenceDataset
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
              # diffusion_model=0,
              # dataset=0,
              # renderer=0,
              ema_decay=0.995,
              train_batch_size=32,
              train_lr=2e-5,
              gradient_accumulate_every=2,
              # step_start_ema=2000, config未涉及项
              # update_ema_every=10,  config未涉及项
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
        # self._cfgs.dd_algo_config=0
        # self._init_dd(self._cfgs.dd_algo_config)
        self._dataset = DeciDiffuserDataset(self._cfgs.train_cfgs.dataset,
                                            batch_size=self._cfgs.algo_cfgs.batch_size,
                                            device=self._device,
                                            horizon=self._cfgs.horizon,
                                            discount=self._cfgs.discount,
                                            returns_scale=self._cfgs.returns_scale,
                                            include_returns=self._cfgs.include_returns,
                                            )
        # dataset = SequenceDataset(
        #     # savepath='dataset_config.pkl',
        #     dataset=self.trans_dataset(),
        #     env_obj=self._env,
        #     env=self._env_id,
        #     horizon=self._cfgs.horizon,
        #     normalizer=self._cfgs.normalizer,
        #     preprocess_fns=self._cfgs.preprocess_fns,
        #     use_padding=self._cfgs.use_padding,
        #     max_path_length=self._cfgs.max_path_length,
        #     include_returns=self._cfgs.include_returns,
        #     returns_scale=self._cfgs.returns_scale,
        #     discount=self._cfgs.discount,
        #     termination_penalty=self._cfgs.termination_penalty,
        # )
        # self._dataset = DDDataLoader(dataset=dataset, train_batch_size=self._cfgs.batch_size, deivce=self._device)
        # self._actor.get_dataset_normalizer(self._dataset.dataset.normalizer)

    def _init_model(self) -> None:
        # 构建GDD模型
        # TU_configs
        # # GDD_configs
        #
        # self.TU=0
        # self.GDD=(self.TU)
        # self.Inv=0
        # self._model=self.GDD

        observation_dim = self._env.observation_space.shape[0]
        action_dim = self._env.action_space.shape[0]
        TUmodel = TemporalUnet(
            # savepath='model_config.pkl',
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
            # savepath='diffusion_config.pkl',
            horizon=self._cfgs.horizon,
            observation_dim=observation_dim,
            action_dim=action_dim,
            n_timesteps=self._cfgs.n_diffusion_steps,
            loss_type=self._cfgs.loss_type,
            clip_denoised=self._cfgs.clip_denoised,
            predict_epsilon=self._cfgs.predict_epsilon,
            ## loss weighting
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
        print(loss)
        # _ = 0

    # def learn(self) -> None:
    #
    #     _ = 0
    # def reset_parameters(self):
    #     self.ema_model.load_state_dict(self.model.state_dict())

    # def step_ema(self):
    #     if self.step < self.step_start_ema:
    #         self.reset_parameters()
    #         return
    #     self.ema.update_model_average(self.ema_model, self.model)

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

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    # def learn(self, n_train_steps):
    #
    #     timer = Timer()
    #     for step in range(n_train_steps):
    #         for i in range(self.gradient_accumulate_every):
    #             batch = next(self.dataloader)
    #             batch = batch_to_device(batch, device=self.device)
    #             loss, infos = self.model.loss(*batch)
    #             loss = loss / self.gradient_accumulate_every
    #             loss.backward()
    #
    #         self.optimizer.step()
    #         self.optimizer.zero_grad()
    #
    #         if self.step % self.update_ema_every == 0:
    #             self.step_ema()
    #
    #         if self.step % self.save_freq == 0:
    #             self.save()

    # if self.step % self.log_freq == 0:
    #     infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
    #     logger.print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}')
    #     metrics = {k:v.detach().item() for k, v in infos.items()}
    #     metrics['steps'] = self.step
    #     metrics['loss'] = loss.detach().item()
    #     logger.log_metrics_summary(metrics, default_stats='mean')

    # if self.step == 0 and self.sample_freq:
    #     self.render_reference(self.n_reference)
    #
    # if self.sample_freq and self.step % self.sample_freq == 0:
    #     if self.model.__class__ == diffuser.models.diffusion.GaussianInvDynDiffusion:
    #         self.inv_render_samples()
    #     elif self.model.__class__ == diffuser.models.diffusion.ActionGaussianDiffusion:
    #         pass
    #     else:
    #         self.render_samples()

    # self.step += 1

    # def save(self):
    #     '''
    #         saves model and ema to disk;
    #         syncs to storage bucket if a bucket is specified
    #     '''
    #     data = {
    #         'step': self.step,
    #         'model': self.model.state_dict(),
    #         'ema': self.ema_model.state_dict()
    #     }
    #     savepath = os.path.join(self.bucket, logger.prefix, 'checkpoint')
    #     os.makedirs(savepath, exist_ok=True)
    #     # logger.save_torch(data, savepath)
    #     if self.save_checkpoints:
    #         savepath = os.path.join(savepath, f'state_{self.step}.pt')
    #     else:
    #         savepath = os.path.join(savepath, 'state.pt')
    #     torch.save(data, savepath)
    #     logger.print(f'[ dd_utils/training ] Saved model to {savepath}')

    # def load(self):
    #     '''
    #         loads model and ema from disk
    #     '''
    #     loadpath = os.path.join(self.bucket, logger.prefix, f'checkpoint/state.pt')
    #     # data = logger.load_torch(loadpath)
    #     data = torch.load(loadpath)
    #
    #     self.step = data['step']
    #     self.model.load_state_dict(data['model'])
    #     self.ema_model.load_state_dict(data['ema'])

    # -----------------------------------------------------------------------------#
    # --------------------------------- rendering ---------------------------------#
    # -----------------------------------------------------------------------------#

    # def render_reference(self, batch_size=10):
    #     '''
    #         renders training points
    #     '''
    #
    #     ## get a temporary dataloader to load a single batch
    #     dataloader_tmp = cycle(torch.dd_utils.data.DataLoader(
    #         self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
    #     ))
    #     batch = dataloader_tmp.__next__()
    #     dataloader_tmp.close()
    #
    #     ## get trajectories and condition at t=0 from batch
    #     trajectories = to_np(batch.trajectories)
    #     conditions = to_np(batch.conditions[0])[:,None]
    #
    #     ## [ batch_size x horizon x observation_dim ]
    #     normed_observations = trajectories[:, :, self.dataset.action_dim:]
    #     observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
    #
    #     # from diffusion.datasets.preprocessing import blocks_cumsum_quat
    #     # # observations = conditions + blocks_cumsum_quat(deltas)
    #     # observations = conditions + deltas.cumsum(axis=1)
    #
    #     #### @TODO: remove block-stacking specific stuff
    #     # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
    #     # observations = blocks_add_kuka(observations)
    #     ####
    #
    #     savepath = os.path.join('images', f'sample-reference.png')
    #     self.renderer.composite(savepath, observations)
    #
    # def render_samples(self, batch_size=2, n_samples=2):
    #     '''
    #         renders samples from (ema) diffusion model
    #     '''
    #     for i in range(batch_size):
    #
    #         ## get a single datapoint
    #         batch = self.dataloader_vis.__next__()
    #         conditions = to_device(batch.conditions, self.device)
    #         ## repeat each item in conditions `n_samples` times
    #         conditions = apply_dict(
    #             einops.repeat,
    #             conditions,
    #             'b d -> (repeat b) d', repeat=n_samples,
    #         )
    #
    #         ## [ n_samples x horizon x (action_dim + observation_dim) ]
    #         if self.ema_model.returns_condition:
    #             returns = to_device(torch.ones(n_samples, 1), self.device)
    #         else:
    #             returns = None
    #
    #         if self.ema_model.model.calc_energy:
    #             samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
    #         else:
    #             samples = self.ema_model.conditional_sample(conditions, returns=returns)
    #
    #         samples = to_np(samples)
    #
    #         ## [ n_samples x horizon x observation_dim ]
    #         normed_observations = samples[:, :, self.dataset.action_dim:]
    #
    #         # [ 1 x 1 x observation_dim ]
    #         normed_conditions = to_np(batch.conditions[0])[:,None]
    #
    #         # from diffusion.datasets.preprocessing import blocks_cumsum_quat
    #         # observations = conditions + blocks_cumsum_quat(deltas)
    #         # observations = conditions + deltas.cumsum(axis=1)
    #
    #         ## [ n_samples x (horizon + 1) x observation_dim ]
    #         normed_observations = np.concatenate([
    #             np.repeat(normed_conditions, n_samples, axis=0),
    #             normed_observations
    #         ], axis=1)
    #
    #         ## [ n_samples x (horizon + 1) x observation_dim ]
    #         observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
    #
    #         #### @TODO: remove block-stacking specific stuff
    #         # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
    #         # observations = blocks_add_kuka(observations)
    #         ####
    #
    #         savepath = os.path.join('images', f'sample-{i}.png')
    #         self.renderer.composite(savepath, observations)
    #
    # def inv_render_samples(self, batch_size=2, n_samples=2):
    #     '''
    #         renders samples from (ema) diffusion model
    #     '''
    #     for i in range(batch_size):
    #
    #         ## get a single datapoint
    #         batch = self.dataloader_vis.__next__()
    #         conditions = to_device(batch.conditions, self.device)
    #         ## repeat each item in conditions `n_samples` times
    #         conditions = apply_dict(
    #             einops.repeat,
    #             conditions,
    #             'b d -> (repeat b) d', repeat=n_samples,
    #         )
    #
    #         ## [ n_samples x horizon x (action_dim + observation_dim) ]
    #         if self.ema_model.returns_condition:
    #             returns = to_device(torch.ones(n_samples, 1), self.device)
    #         else:
    #             returns = None
    #
    #         if self.ema_model.model.calc_energy:
    #             samples = self.ema_model.grad_conditional_sample(conditions, returns=returns)
    #         else:
    #             samples = self.ema_model.conditional_sample(conditions, returns=returns)
    #
    #         samples = to_np(samples)
    #
    #         ## [ n_samples x horizon x observation_dim ]
    #         normed_observations = samples[:, :, :]
    #
    #         # [ 1 x 1 x observation_dim ]
    #         normed_conditions = to_np(batch.conditions[0])[:,None]
    #
    #         # from diffusion.datasets.preprocessing import blocks_cumsum_quat
    #         # observations = conditions + blocks_cumsum_quat(deltas)
    #         # observations = conditions + deltas.cumsum(axis=1)
    #
    #         ## [ n_samples x (horizon + 1) x observation_dim ]
    #         normed_observations = np.concatenate([
    #             np.repeat(normed_conditions, n_samples, axis=0),
    #             normed_observations
    #         ], axis=1)
    #
    #         ## [ n_samples x (horizon + 1) x observation_dim ]
    #         observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')
    #
    #         #### @TODO: remove block-stacking specific stuff
    #         # from diffusion.datasets.preprocessing import blocks_euler_to_quat, blocks_add_kuka
    #         # observations = blocks_add_kuka(observations)
    #         ####
    #
    #         savepath = os.path.join('images', f'sample-{i}.png')
    #         self.renderer.composite(savepath, observations)
