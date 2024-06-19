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
"""Implementation of the TRPO algorithm with Control Barrier Function."""
# mypy: ignore-errors

from __future__ import annotations

import os

import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.adapter.barrier_function_adapter import BarrierFunctionAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.common.barrier_comp import BarrierCompensator
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.typing import Any
from omnisafe.utils import distributed
from omnisafe.utils.distributed import get_rank


@registry.register
class TRPOCBF(TRPO):
    """The TRPO algorithm with CBF.

    References:
        - Title: End-to-end safe reinforcement learning through barrier functions for
        safety-critical continuous control tasks
        - Authors: R Cheng, G Orosz, RM Murray, JW Burdick.
        - URL: `TRPOCBF <https://ojs.aaai.org/index.php/AAAI/article/view/4213/4091>`_
    """

    def _init_log(self) -> None:
        """Log the TRPOCBF specific information.

        +----------------------------+---------------------------------+
        | Things to log              | Description                     |
        +============================+=================================+
        | Value/Loss_compensator     | The Loss of action compensator. |
        +----------------------------+---------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Value/Loss_compensator')

    def _init_env(self) -> None:
        self._env: BarrierFunctionAdapter = BarrierFunctionAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )
        self.solver = PendulumSolver(device=self._device)
        self.compensator = BarrierCompensator(
            obs_dim=self._env.observation_space.shape[0],
            act_dim=self._env.action_space.shape[0],
            cfgs=self._cfgs.compensator_cfgs,
        ).to(self._device)
        self._env.set_solver(solver=self.solver)
        self._env.set_compensator(compensator=self.compensator)

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

    def _update(self) -> None:
        """Update actor, critic.

        .. hint::
            Here are some differences between NPG and Policy Gradient (PG): In PG, the actor network
            and the critic network are updated together. When the KL divergence between the old
            policy, and the new policy is larger than a threshold, the update is rejected together.

            In NPG, the actor network and the critic network are updated separately. When the KL
            divergence between the old policy, and the new policy is larger than a threshold, the
            update of the actor network is rejected, but the update of the critic network is still
            accepted.
        """
        data = self._buf.get()

        (
            obs,
            act,
            logp,
            target_value_r,
            adv_r,
            adv_c,
            approx_compensating_act,
            compensating_act,
        ) = (
            data['obs'],
            data['act'],
            data['logp'],
            data['target_value_r'],
            data['adv_r'],
            data['adv_c'],
            data['approx_compensating_act'],
            data['compensating_act'],
        )

        self._update_actor(obs, act, logp, adv_r, adv_c)
        compensator_loss = self._env.compensator.update(
            observation=obs,
            approx_compensating_act=approx_compensating_act,
            compensating_act=compensating_act,
        )
        dataloader = DataLoader(
            dataset=TensorDataset(obs, target_value_r),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        for _ in range(self._cfgs.algo_cfgs.update_iters):
            for (
                obs,
                target_value_r,
            ) in dataloader:
                self._update_reward_critic(obs, target_value_r)

        self._logger.store(
            {
                'Train/StopIter': self._cfgs.algo_cfgs.update_iters,
                'Value/Adv': adv_r.mean().item(),
                'Value/Loss_compensator': compensator_loss.item(),
            },
        )

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
