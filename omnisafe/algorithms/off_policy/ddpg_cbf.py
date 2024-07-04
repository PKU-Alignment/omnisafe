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
"""Implementation of the DDPG algorithm with Control Barrier Function."""
# mypy: ignore-errors


from __future__ import annotations

import os

import joblib

from omnisafe.adapter.offpolicy_barrier_function_adapter import OffPolicyBarrierFunctionAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.typing import Any
from omnisafe.utils.distributed import get_rank


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class DDPGCBF(DDPG):
    """The DDPG algorithm with CBF.

    References:
        - Title: End-to-end safe reinforcement learning through barrier functions for
        safety-critical continuous control tasks
        - Authors: R Cheng, G Orosz, RM Murray, JW Burdick.
        - URL: `DDPGCBF <https://ojs.aaai.org/index.php/AAAI/article/view/4213/4091>`_
    """

    def _init_env(self) -> None:
        super()._init_env()
        self._env: OffPolicyBarrierFunctionAdapter = OffPolicyBarrierFunctionAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )

    def _init_log(self) -> None:
        """Log the DDPGCBF specific information.

        +----------------------------+---------------------------------+
        | Things to log              | Description                     |
        +============================+=================================+
        | Value/Loss_compensator     | The Loss of action compensator. |
        +----------------------------+---------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Value/Loss_compensator')

    def _specific_save(self) -> None:
        """Save some algorithms specific models per epoch."""
        super()._specific_save()
        if get_rank() == 0:
            path = os.path.join(
                self._logger.log_dir,
                'gp_model_save',
                f'gaussian_process_regressor_{self._logger.current_epoch}.pkl',
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(self._env.gp_models, path)

    def _setup_torch_saver(self) -> None:
        """Define what need to be saved below.

        OmniSafe's main storage interface is based on PyTorch. If you need to save models in other
        formats, please use :meth:`_specific_save`.
        """
        what_to_save: dict[str, Any] = {}

        what_to_save['pi'] = self._actor_critic.actor
        what_to_save['compensator'] = self._env.compensator
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
