# Copyright 2022 OmniSafe Team. All Rights Reserved.
# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Implementation of the Saute algorithm."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO


@registry.register
class PPOSaute(PPO):
    """Saute algorithm implemented by PPO.

    References:
        Paper Name: Saute RL: Almost Surely Safe Reinforcement Learning Using State Augmentation.
        Paper author: Aivar Sootla, Alexander I. Cowen-Rivers, Taher Jafferjee, Ziyan Wang,
                      David Mguni, Jun Wang, Haitham Bou-Ammar.
        Paper URL: https://arxiv.org/abs/2202.06558
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env_id,
        cfgs,
        algo='ppo_saute',
        clip=0.2,
        wrapper_type: str = 'SauteEnvWrapper',
    ) -> None:
        r"""Initialize PPOSaute."""
        self.clip = clip
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
            algo=algo,
            wrapper_type=wrapper_type,
        )

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Metrics/EpBudget')
