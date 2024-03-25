# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Implementation of Decision Diffuser."""

import copy
import time
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.sequence_dataset import RewardBatch, SequenceDataset
from omnisafe.models.actor import DecisionDiffuserActor
from omnisafe.utils.config import Config
from omnisafe.utils.ema import EMA


@registry.register
class DecisionDiffuser(BaseOffline):
    """Decision Diffuser algorithm.
        The Decision Diffuser is an approach to decision-making that uses 
        conditional generative modeling, specifically diffusion models, 
        instead of traditional reinforcement learning (RL). It generates 
        actions that achieve desired outcomes. By conditioning on factors 
        like outcomes, constraints, and skills, it effectively generates 
        adaptive and complex behaviors. 

    References:
        - Title: Is conditional generative modeling all you need for decision-making?
        - Author: Anurag Ajay, Yilun Du, Abhi Gupta, Joshua B. Tenenbaum, Tommi Jaakkola, Pulkit Agrawal
        - URL: https://arxiv.org/pdf/2211.15657.pdf
    """

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of :class:`BaseOffline`."""
        super().__init__(env_id, cfgs)

        self._actor: DecisionDiffuserActor
        self.epoch: int = 0
        self._step: int = 0

    def _init_log(self) -> None:
        """Log the Decision Diffuser specific information.

        +-------------------------+----------------------------------------------------+
        | Things to log           | Description                                        |
        +=========================+====================================================+
        | Loss/Loss               | Loss of Diffusion and InvAR network                |
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
        self._step = 0

        def data_iter(batch_size: int, x: SequenceDataset) -> Iterator[RewardBatch]:
            num_examples = len(x)
            indices = list(range(num_examples))
            np.random.shuffle(indices)
            for i in range(0, num_examples, batch_size):
                batch_indices = indices[i : min(i + batch_size, num_examples)]
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

        self._opt = torch.optim.Adam(
            self._actor.parameters(),
            lr=self._cfgs.model_cfgs.learning_rate,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self._actor.load_state_dict(self._learn_model.state_dict())

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
        cls_free_conditions: torch.Tensor,
    ) -> None:
        # pylint: disable=arguments-differ
        loss = self._learn_model.loss(x, state_condition, cls_free_conditions)

        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        if self._step % self._update_ema_every == 0:
            self._step_ema()
        self._step += 1

        self._logger.store(
            **{
                'Loss/Loss': loss.item(),
            },
        )
