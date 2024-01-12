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

import copy
import time
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.sequence_dataset import SequenceDataset  # type: ignore
from omnisafe.models.actor import DecisionDiffuserActor
from omnisafe.utils.ema import EMA


@registry.register
class DecisionDiffuser(BaseOffline):
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
        | Loss/Loss           | Loss of Diffusion and InvAR network               |
        +-------------------------+----------------------------------------------------+
        | Loss/Loss_A0         | Loss of Advantage                 |
        +-------------------------+----------------------------------------------------+
        """
        super()._init_log()
        what_to_save: Dict[str, Any] = {
            'diffuser': self._actor,
            'learn_model': self._learn_model,
        }
        self._logger.setup_torch_saver(what_to_save)

        self._logger.register_key('Loss/Loss')
        self._logger.register_key('Loss/Loss_A0')

    def _init(self) -> None:
        self._seq_dataset = SequenceDataset(
            self._cfgs.train_cfgs.dataset,
            horizon=self._cfgs.model_cfgs.horizon,
            max_n_episodes=self._cfgs.train_cfgs.max_n_episodes,
            max_path_length=self._cfgs.train_cfgs.max_path_length,
            reward_discount=self._cfgs.train_cfgs.discount,
        )

        def data_iter(batch_size: int, x: torch.utils.data.Dataset) -> Iterator[torch.Tensor]:
            num_examples = len(x)  # type: ignore
            indices = list(range(num_examples))
            np.random.shuffle(indices)
            for i in range(0, num_examples, batch_size):
                batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)])
                yield x[batch_indices]

        self._dataloader = data_iter(self._cfgs.algo_cfgs.batch_size, self._seq_dataset)

    def _init_model(self) -> None:
        self._actor = DecisionDiffuserActor(
            horizon=80,
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            cls_free_cond_dim=self._cfgs.model_cfgs.cls_free_cond_dim,
        ).to(self._device)
        # use ema model for more stable training
        self._learn_model: DecisionDiffuserActor = copy.deepcopy(self._actor)

        self._ema = EMA(self._cfgs.train_cfgs.ema_decay)
        self._update_ema_every = self._cfgs.train_cfgs.update_ema_every
        self._step_start_ema = self._cfgs.train_cfgs.step_start_ema

        self._denoise_opt = torch.optim.Adam(
            self._actor._model.model.parameters(),
            lr=self._cfgs.model_cfgs.learning_rate,
        )
        self._act_opt = torch.optim.Adam(
            self._actor._model.inv_model.parameters(),
            lr=self._cfgs.model_cfgs.inv_ar_learning_rate,
        )

        self._reset_parameters()
        self._step = 0

    def _reset_parameters(self) -> None:
        self._actor._model.load_state_dict(self._learn_model._model.state_dict())

    def _step_ema(self) -> None:
        if self._step < self._step_start_ema:
            self._reset_parameters()
            return
        self._ema.update_model_average(self._actor, self._learn_model)

    def learn(self) -> Tuple[float, float, float]:
        """Learn the policy."""
        self._logger.log('Start training ...')
        self._logger.torch_save()

        start_time = time.time()
        epoch_time = time.time()

        for step in range(self._cfgs.train_cfgs.total_steps):
            self._step = step
            batch = next(self._dataloader)
            trajectories, state_condition, cls_free_conditions = batch
            self._train(trajectories, state_condition, cls_free_conditions)
            self._step_ema()

            if (step + 1) % self._cfgs.algo_cfgs.steps_per_epoch == 0:
                self.epoch = (step + 1) // self._cfgs.algo_cfgs.steps_per_epoch
                self._logger.store(**{'Time/Update': time.time() - epoch_time})
                eval_time = time.time()
                # self._evaluate()

                self._logger.store(
                    {
                        'Time/Evaluate': time.time() - eval_time,
                        'Time/Epoch': time.time() - epoch_time,
                        'Time/Total': time.time() - start_time,
                        'Train/Epoch': self.epoch,
                        'TotalSteps': step + 1,
                    },
                )

                epoch_time = time.time()
                self._logger.dump_tabular()

                # save model to disk
                if self.epoch % self._cfgs.logger_cfgs.save_model_freq == 0:
                    self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _train(  # type: ignore
        self,
        x: torch.Tensor,
        state_condition: Dict[int, torch.Tensor],
        cls_free_conditions: List[torch.Tensor],
    ) -> None:
        loss, infos = self._learn_model._model.loss(x, state_condition, cls_free_conditions)

        self._denoise_opt.zero_grad()
        self._act_opt.zero_grad()
        loss.backward()
        self._denoise_opt.step()
        self._act_opt.step()

        if self._step % self._update_ema_every == 0:
            self._step_ema()
        self._step += 1

        self._logger.store(
            **{
                'Loss/Loss': loss.item(),
                'Loss/Loss_A0': infos['a0_loss'].item(),
            },
        )
