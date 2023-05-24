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
"""Implementation of the Safe Learning Off-Policy with Online Planning algorithm."""


from __future__ import annotations

from gymnasium.spaces import Box

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.base.ensemble import EnsembleDynamicsModel
from omnisafe.algorithms.model_based.base.loop import LOOP
from omnisafe.algorithms.model_based.planner.safe_arc import SafeARCPlanner
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.typing import OmnisafeSpace
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes, too-few-public-methods
class SafeLOOP(LOOP):
    """The Safe Learning Off-Policy with Online Planning (SafeLOOP) algorithm.

    References:
        - Title: Learning Off-Policy with Online Planning
        - Authors: Harshit Sikchi, Wenxuan Zhou, David Held.
        - URL: `SafeLOOP <https://arxiv.org/abs/2008.10066>`_
    """

    def _init_model(self) -> None:
        """Initialize the dynamics model and the planner.

        SafeLOOP uses following models:

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
        self._actor_critic: ConstraintActorQCritic = ConstraintActorQCritic(
            obs_space=self._dynamics_state_space,
            act_space=self._action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._epochs,
        ).to(self._device)
        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)
        self._use_actor_critic: bool = True
        self._update_count: int = 0
        self._dynamics: EnsembleDynamicsModel = EnsembleDynamicsModel(
            model_cfgs=self._cfgs.dynamics_cfgs,
            device=self._device,
            state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            actor_critic=self._actor_critic,
            rew_func=None,
            cost_func=self._env.get_cost_from_obs_tensor,
            terminal_func=None,
        )
        self._update_dynamics_cycle: int = int(self._cfgs.algo_cfgs.update_dynamics_cycle)

        self._planner: SafeARCPlanner = SafeARCPlanner(
            dynamics=self._dynamics,
            planner_cfgs=self._cfgs.planner_cfgs,
            gamma=float(self._cfgs.algo_cfgs.gamma),
            cost_gamma=float(self._cfgs.algo_cfgs.cost_gamma),
            dynamics_state_shape=self._dynamics_state_space.shape,
            action_shape=self._action_space.shape,
            action_max=1.0,
            action_min=-1.0,
            device=self._device,
            cost_limit=float(self._cfgs.algo_cfgs.cost_limit),
            actor_critic=self._actor_critic,
        )

    def _init_log(self) -> None:
        """Initialize the logger keys for the algorithm.

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
