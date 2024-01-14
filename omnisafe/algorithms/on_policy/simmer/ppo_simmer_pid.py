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
"""Implementation of the Simmer version of the PPO algorithm."""

import torch

from omnisafe.adapter.simmer_adapter import SimmerAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed


@registry.register
class PPOSimmerPID(PPO):
    """The Simmer version(based on PID controller) of the PPO algorithm.

    A simple combination of the Simmer RL and the Proximal Policy Optimization algorithm.

    References:
        - Title: Effects of Safety State Augmentation on Safe Exploration.
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Jun Wang, Haitham Bou Ammar.
        - URL: `PPOSimmerPID <https://arxiv.org/pdf/2206.02675.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.SimmerAdapter` to adapt the environment to the algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()
        """
        self._env: SimmerAdapter = SimmerAdapter(
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

    def _init_log(self) -> None:
        """Log the PPOSimmerPID specific information.

        +------------------+-----------------------------------+
        | Things to log    | Description                       |
        +==================+===================================+
        | Metrics/EpBudget | The safety budget of the episode. |
        +------------------+-----------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/EpBudget')

    def _update(self) -> None:
        """Update actor, critic, as we used in the :class:`PolicyGradient` algorithm."""
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        self._env.control_budget(torch.as_tensor(Jc, dtype=torch.float32, device=self._device))
        super()._update()
