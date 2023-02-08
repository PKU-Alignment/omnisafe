# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Lagrange version of the Saute algorithm using PPOLag."""

from typing import NamedTuple

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag


@registry.register
class PPOLagSaute(PPOLag):
    """The Saute algorithm implemented with PPOLag.

    References:
        - Title: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation
        - Authors: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee,
            Ziyan Wang, David Mguni, Jun Wang, Haitham Bou-Ammar.
        - URL: `Saute RL<https://arxiv.org/abs/2202.06558>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize PPOLagSaute.

        PPOLagSaute is a combination of :class:`PPO` and :class:`Lagrange` model,
        using :class:`Saute` as the environment wrapper.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        super().__init__(env_id=env_id, cfgs=cfgs)

    def algorithm_specific_logs(self) -> None:
        """Log the Saute specific information.

        .. list-table::

            *   -   Things to log
                -   Description
            *   -   Metrics/EpBudget
                -   The budget of the episode.
        """
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/EpBudget')
