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

from typing import Any, Dict, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.offline.base import BaseOffline
from omnisafe.common.offline.dataset import DeciDiffuserDataset
from omnisafe.models.actor import ActorBuilder


# pylint: disable=C
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
            'pi': self._actor,
        }
        self._logger.setup_torch_saver(what_to_save)
        self._logger.register_key('Loss/Loss_diffuser')
        self._logger.register_key('Loss/Loss_inv')
        self._logger.register_key('Loss/Loss_total')

    def _init(self) -> None:
        self._dataset = DeciDiffuserDataset(
            self._cfgs.train_cfgs.dataset,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            device=self._device,
            horizon=self._cfgs.algo_cfgs.horizon,
            discount=self._cfgs.algo_cfgs.gamma,
            returns_scale=self._cfgs.dataset_cfgs.returns_scale,
            include_returns=self._cfgs.dataset_cfgs.include_returns,
            include_constraints=self._cfgs.dataset_cfgs.include_constraints,
            include_skills=self._cfgs.dataset_cfgs.include_skills,
        )

    # def _init_env(self) -> None:
    #     self._env = DecisionDiffuserAdpater(self._env_id, self._seed, self._cfgs)

    def _init_model(self) -> None:
        self._actor = (
            ActorBuilder(
                obs_space=self._env.observation_space,
                act_space=self._env.action_space,
                hidden_sizes=[],
                custom_cfgs=self._cfgs,
            )
            .build_actor('decisiondiffuser')
            .to(self._device)
        )

        self._optimizer = torch.optim.Adam(self._actor.parameters(), lr=self._cfgs.model_cfgs.lr)

    def _train(
        self,
        batch: Tuple[torch.Tensor, ...],
    ) -> None:
        for _i in range(self._cfgs.train_cfgs.gradient_accumulate_every):
            loss, infos = self._actor.loss(*batch)
            loss = loss / self._cfgs.train_cfgs.gradient_accumulate_every
            loss.backward()

        self._logger.store(
            **{
                'Loss/Loss_diffuser': infos['loss_diffuser'].item(),
                'Loss/Loss_inv': infos['loss_inv'].item(),
                'Loss/Loss_total': infos['loss_total'].item(),
            },
        )
        self._optimizer.step()
        self._optimizer.zero_grad()
