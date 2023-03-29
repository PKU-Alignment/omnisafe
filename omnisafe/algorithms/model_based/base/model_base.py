

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
"""Implementation of the Deep Deterministic Policy Gradient algorithm."""

import time
from typing import Any, Dict, Tuple, Union, Optional


import torch
from torch import nn

from abc import ABC, abstractmethod
from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class ModelBase(BaseAlgo,ABC):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def learn(self) -> Tuple[Union[int, float], ...]:
        """This is main function for algorithm update, divided into the following steps:

        - :meth:`rollout`: collect interactive data from environment.
        - :meth:`update`: perform actor/critic updates.
        - :meth:`log`: epoch/update information for visualization and terminal log print.
        """
        self._logger.log('INFO: Start training')
        start_time = time.time()
        total_steps = 0
        ep_len, ep_ret, ep_cost = 0,0,0
        for epoch in range(self._epochs):
            roll_out_time = 0.0
            epoch_start_time = time.time()
            update_actor_critic_time = 0
            update_dynamics_time = 0
            epoch_steps = 0
            while epoch_steps < self._steps_per_epoch:
                action, action_info = self.select_action(state)
                next_state, reward, cost, terminated, truncated, info = self.env.step(
                    action, self.cfgs.action_repeat
                )
                epoch_steps += info['step_num']
                total_steps += info['step_num']
                ep_len += info['step_num']

                ep_cost += (self.cost_gamma**ep_len) * cost
                ep_ret += reward

                self.store_real_data(
                    total_steps,
                    ep_len,
                    state,
                    action_info,
                    action,
                    reward,
                    cost,
                    terminated,
                    truncated,
                    next_state,
                    info,
                )
                state = next_state
                if terminated or truncated:
                    self.logger.store(
                        **{
                            'Metrics/EpRet': ep_ret,
                            'Metrics/EpLen': ep_len * self.cfgs.action_repeat,
                            'Metrics/EpCost': ep_cost,
                        }
                    )
                    ep_ret, ep_cost, ep_len = 0, 0, 0
                    state = self.env.reset()
                    self.algo_reset()
                if (
                    total_steps % self.cfgs.update_dynamics_freq < self.cfgs.action_repeat
                    and total_steps - last_dynamics_update >= self.cfgs.update_dynamics_freq
                ):
                    update_dynamics_start = time.time()
                    self.update_dynamics_model()
                    last_dynamics_update = total_steps
                    update_dynamics_time += time.time() - update_dynamics_start

                if (
                    self._use_actor_critic
                    and total_steps % self.cfgs.update_policy_freq < self.cfgs.action_repeat
                    and total_steps - last_policy_update >= self.cfgs.update_policy_freq
                ):
                    update_actor_critic_start = time.time()
                    self.update_actor_critic(total_steps)
                    last_policy_update = total_steps
                    update_actor_critic_time += time.time() - update_actor_critic_start
            epoch_time = time.time() - epoch_start_time
            # Evaluate episode
            self._logger.store(
                **{
                    'Train/Epoch': epoch,
                    'TotalEnvSteps': total_steps,
                    #'Time/FPS': self._cfgs.algo_cfgs.update_cycle / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': epoch_time,
                    #'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                }
            )
            self._logger.store(**{'Time/UpdateDynamics': update_dynamics_time})
            self._logger.store(**{'Time/UpdateActor': update_actor_critic_time})
            roll_out_time = epoch_time - update_dynamics_time - update_actor_critic_time
            self._logger.store(**{'Time/Rollout': roll_out_time})
            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = self._logger.get_stats('Metrics/EpLen')[0]
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    @property
    @abstractmethod
    def update_dynamics_model(self):
        """Update dynamics."""

    @property
    @abstractmethod
    def select_action(self, state, if_training=True):
        """action selection"""






