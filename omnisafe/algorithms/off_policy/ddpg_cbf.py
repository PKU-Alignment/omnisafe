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
"""Implementation of the DDPG algorithm with Control Barrier Function."""
# mypy: ignore-errors


from __future__ import annotations

import os

import joblib
import torch

from omnisafe.adapter.offpolicy_barrier_function_adapter import OffPolicyBarrierFunctionAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.common.barrier_solver import PendulumSolver
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
        self._env: OffPolicyBarrierFunctionAdapter = OffPolicyBarrierFunctionAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        solver = PendulumSolver(device=self._cfgs.train_cfgs.device)
        compensator = BarrierCompensator(
            obs_dim=self._env.observation_space.shape[0],
            act_dim=self._env.action_space.shape[0],
            cfgs=self._cfgs.compensator_cfgs,
        ).to(self._device)

        self._env.set_compensator(compensator=compensator)
        self._env.set_solver(solver=solver)

        assert (
            self._cfgs.algo_cfgs.steps_per_epoch % self._cfgs.train_cfgs.vector_env_nums == 0
        ), 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch // self._cfgs.train_cfgs.vector_env_nums
        )

        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init(self) -> None:
        super()._init()
        self._buf.add_field(
            name='approx_compensating_act',
            shape=self._env.action_space.shape,
            dtype=torch.float32,
        )
        self._buf.add_field(
            name='compensating_act',
            shape=self._env.action_space.shape,
            dtype=torch.float32,
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

    def _log_what_to_save(self) -> dict[str, Any]:
        """Define what need to be saved below."""
        what_to_save: dict[str, Any] = {}

        what_to_save['pi'] = self._actor_critic.actor
        what_to_save['compensator'] = self._env.compensator
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
