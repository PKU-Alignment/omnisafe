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
"""Implementation of a basic algorithm framework for offline algorithms."""

import time
from abc import abstractmethod
from typing import Tuple, Union

import torch

from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.logger import Logger
from omnisafe.common.offline.dataset import OfflineDataset
from omnisafe.utils.config import Config


class BaseOffline(BaseAlgo):
    """Base class for offline algorithms."""

    def __init__(self, env_id: str, cfgs: Config) -> None:
        self._logger: Logger
        self._dataset: OfflineDataset

        super().__init__(env_id, cfgs)

    def learn(self) -> Tuple[Union[int, float], ...]:
        self._logger.log('Start training ...')

        start_time = time.time()
        epoch_time = time.time()

        for step in range(self._cfgs.train_cfgs.num_steps):
            batch = self._dataset.sample()
            self._train(batch)

            if (step + 1) % self._cfgs.steps_per_epoch == 0:
                epoch = (step + 1) // self._cfgs.steps_per_epoch
                self._logger.store(**{'Time/Update': time.time() - epoch_time})
                evla_time = time.time()
                self._evaluate()

                self._logger.store(
                    **{
                        'Time/Evaluate': time.time() - evla_time,
                        'Time/Epoch': time.time() - epoch_time,
                        'Time/Total': time.time() - start_time,
                        'Train/Epoch': epoch,
                        'TotalSteps': step + 1,
                    }
                )

                epoch_time = time.time()
                self._logger.dump_tabular()

                # save model to disk
                if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                    self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    @abstractmethod
    def _train(
        self,
        batch: Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
        ],
    ):
        """Train the model."""

    def _evaluate(self) -> None:
        """Evaluate the model."""
        self._logger.log('Start evaluation ...')
        self._logger.log('Evaluation finished.')
