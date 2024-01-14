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
"""Implementation of the Constrained Cross-Entropy algorithm."""


from __future__ import annotations

from gymnasium.spaces import Box

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.base import PETS
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.planner.cce import CCEPlanner
from omnisafe.typing import OmnisafeSpace


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class CCEPETS(PETS):
    """The Constrained Cross-Entropy (CCE) algorithm implementation based on PETS.

    References:
        - Title: Constrained Cross-Entropy Method for Safe Reinforcement Learning
        - Authors: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
            Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        - URL: `CCE <https://proceedings.neurips.cc/paper/2018/hash/34ffeb359a192eb8174b6854643cc046-Abstract.html>`_
    """

    def _init_model(self) -> None:
        """Initialize the dynamics model and the planner.

        CCEPETS uses following models:

        - dynamics model: to predict the next state and the cost.
        - planner: to generate the action.
        """
        self._dynamics_state_space: OmnisafeSpace = (
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

        self._planner: CCEPlanner = CCEPlanner(
            dynamics=self._dynamics,
            planner_cfgs=self._cfgs.planner_cfgs,
            gamma=float(self._cfgs.algo_cfgs.gamma),
            cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device=self._device,
            cost_limit=self._cfgs.algo_cfgs.cost_limit,
        )

        self._use_actor_critic: bool = False
        self._update_dynamics_cycle: int = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

    def _init_log(self) -> None:
        """Initialize the logger keys for the CCE algorithm.

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
        """
        super()._init_log()
        self._logger.register_key('Plan/feasible_num')
        self._logger.register_key('Plan/episode_costs_max')
        self._logger.register_key('Plan/episode_costs_mean')
        self._logger.register_key('Plan/episode_costs_min')
