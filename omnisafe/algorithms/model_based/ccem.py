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

from omnisafe.adapter import ModelBasedAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import OffPolicyBuffer
from omnisafe.common.logger import Logger

from omnisafe.algorithms.model_based.models import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.cce import CCEPlanner
from omnisafe.algorithms.model_based.base import PETS
import numpy as np
from matplotlib import pylab
from gymnasium.utils.save_video import save_video
import os


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class CCEM(PETS):
    """The Deep Deterministic Policy Gradient (DDPG) algorithm.

    References:

        - Title: Continuous control with deep reinforcement learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
        Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `DDPG <https://arxiv.org/abs/1509.02971>`_
    """

    def _init_model(self) -> None:
        self._dynamics_state_space = self._env.coordinate_observation_space if self._env.coordinate_observation_space is not None else self._env.observation_space
        self._dynamics = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_size=self._dynamics_state_space.shape[0],
            action_size=self._env.action_space.shape[0],
            reward_size=1,
            cost_size=1,
            use_cost=True,
            use_terminal=False,
            use_var=False,
            use_reward_critic=False,
            use_cost_critic=False,
            actor_critic=None,
            rew_func=None,
            cost_func=self._env.get_cost_from_obs_tensor,
            terminal_func=None,
        )

        self._planner = CCEPlanner(
            dynamics=self._dynamics,
            num_models=self._cfgs.dynamics_cfgs.num_ensemble,
            horizon=self._cfgs.algo_cfgs.plan_horizon,
            num_iterations=self._cfgs.algo_cfgs.num_iterations,
            num_particles=self._cfgs.algo_cfgs.num_particles,
            num_samples=self._cfgs.algo_cfgs.num_samples,
            num_elites=self._cfgs.algo_cfgs.num_elites,
            momentum=self._cfgs.algo_cfgs.momentum,
            epsilon=self._cfgs.algo_cfgs.epsilon,
            gamma=self._cfgs.algo_cfgs.gamma,
            cost_gamma=self._cfgs.algo_cfgs.cost_gamma,
            cost_limit=self._cfgs.algo_cfgs.cost_limit,
            device=self._device,
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._env.action_space.shape,
            action_max=1.0,
            action_min=-1.0,
        )

        self._use_actor_critic = False
        self._update_dynamics_cycle = int(self._cfgs.algo_cfgs.update_dynamics_cycle)


    def _init_log(self) -> None:
        super()._init_log()
        self._logger.register_key('Plan/feasible_num')
        self._logger.register_key('Plan/episode_costs_max')
        self._logger.register_key('Plan/episode_costs_mean')
        self._logger.register_key('Plan/episode_costs_min')

    def _select_action(
            self,
            current_step: int,
            state: torch.Tensor,
            env: ModelBasedAdapter,
            ) -> Tuple[np.ndarray, Dict]:
        """action selection"""
        if current_step < self._cfgs.algo_cfgs.start_learning_steps:
            action = torch.tensor(self._env.action_space.sample()).to(self._device).unsqueeze(0)
            #action = torch.rand(size=1, *self._env.action_space.shape)
        else:
            action, info = self._planner.output_action(state)
            #action = action.cpu().detach().numpy()
            self._logger.store(
                **{
                'Plan/iter': info['Plan/iter'],
                'Plan/last_var_max': info['Plan/last_var_max'],
                'Plan/last_var_mean': info['Plan/last_var_mean'],
                'Plan/last_var_min': info['Plan/last_var_min'],
                'Plan/feasible_num': info['Plan/feasible_num'],
                'Plan/episode_returns_max': info['Plan/episode_returns_max'],
                'Plan/episode_returns_mean': info['Plan/episode_returns_mean'],
                'Plan/episode_returns_min': info['Plan/episode_returns_min'],
                'Plan/episode_costs_max': info['Plan/episode_costs_max'],
                'Plan/episode_costs_mean': info['Plan/episode_costs_mean'],
                'Plan/episode_costs_min': info['Plan/episode_costs_min'],
                }
            )
        assert action.shape == torch.Size([state.shape[0], self._env.action_space.shape[0]]), "action shape should be [batch_size, action_dim]"
        info = {}
        return action, info






