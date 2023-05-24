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
"""Implementation of the Conservative and Adaptive Penalty algorithm."""


from __future__ import annotations

from typing import Any

import numpy as np
from gymnasium.spaces import Box

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.base import PETS
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.cap import CAPPlanner
from omnisafe.common.lagrange import Lagrange


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class CAPPETS(PETS):
    """The Conservative and Adaptive Penalty (CAP) algorithm implementation based on PETS.

    References:
        - Title: Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning
        - Authors: Yecheng Jason Ma, Andrew Shen, Osbert Bastani, Dinesh Jayaraman.
        - URL: `CAP <https://arxiv.org/abs/2112.07701>`_
    """

    def _init_model(self) -> None:
        """Initialize the dynamics model and the planner.

        CAP uses following models:

        - dynamics model: to predict the next state and the cost.
        - lagrange multiplier: to trade off between the cost and the reward.
        - planner: to generate the action.
        """
        self._dynamics_state_space = (
            self._env.coordinate_observation_space
            if self._env.coordinate_observation_space is not None
            else self._env.observation_space
        )
        assert self._dynamics_state_space is not None and isinstance(
            self._dynamics_state_space.shape,
            tuple,
        )
        assert self._env.action_space is not None and isinstance(
            self._env.action_space.shape,
            tuple,
        )
        if isinstance(self._env.action_space, Box):
            self._action_space = self._env.action_space
        else:
            raise NotImplementedError

        self._dynamics: EnsembleDynamicsModel = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            actor_critic=None,
            rew_func=None,
            cost_func=self._env.get_cost_from_obs_tensor,
            terminal_func=None,
        )

        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

        self._planner: CAPPlanner = CAPPlanner(
            dynamics=self._dynamics,
            planner_cfgs=self._cfgs.planner_cfgs,
            gamma=float(self._cfgs.algo_cfgs.gamma),
            cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device=self._device,
            cost_limit=self._cfgs.lagrange_cfgs.cost_limit,
            lagrange=self._lagrange.lagrangian_multiplier,
        )

        self._use_actor_critic: bool = False
        self._update_dynamics_cycle: int = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

    def _init_log(self) -> None:
        """Initialize the logger.

        +----------------------------+-------------------------------+
        | Things to log              | Description                   |
        +============================+===============================+
        | Plan/feasible_num          | The number of feasible plans. |
        +----------------------------+-------------------------------+
        | Plan/episode_costs_max     | The maximum planning cost.    |
        +----------------------------+-------------------------------+
        | Plan/episode_costs_mean    | The mean planning cost.       |
        +----------------------------+-------------------------------+
        | Plan/episode_costs_min     | The minimum planning cost.    |
        +----------------------------+-------------------------------+
        | Metrics/LagrangeMultiplier | The lagrange multiplier.      |
        +----------------------------+-------------------------------+
        | Plan/var_penalty_max       | The maximum planning penalty. |
        +----------------------------+-------------------------------+
        | Plan/var_penalty_mean      | The mean planning penalty.    |
        +----------------------------+-------------------------------+
        | Plan/var_penalty_min       | The minimum planning penalty. |
        +----------------------------+-------------------------------+

        """
        super()._init_log()
        self._logger.register_key('Plan/feasible_num')
        self._logger.register_key('Plan/episode_costs_max')
        self._logger.register_key('Plan/episode_costs_mean')
        self._logger.register_key('Plan/episode_costs_min')
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Plan/var_penalty_max')
        self._logger.register_key('Plan/var_penalty_mean')
        self._logger.register_key('Plan/var_penalty_min')

    def _save_model(self) -> None:
        """Save the model."""
        what_to_save: dict[str, Any] = {}
        # set up model saving
        what_to_save = {
            'dynamics': self._dynamics.ensemble_model,
            'lagrangian_multiplier': self._lagrange.lagrangian_multiplier,
        }
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        self._logger.setup_torch_saver(what_to_save)
        # self._logger.planner_save()
        self._logger.torch_save()

    def _update_epoch(self) -> None:
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        self._logger.store(**{'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})
