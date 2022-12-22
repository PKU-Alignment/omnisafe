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
"""Implementation of the Early terminated algorithm by PPOLag."""

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag


@registry.register
class PPOLagEarlyTerminated(PPOLag):
    r"""Early terminated algorithm implemented by PPOLag.

    References:
        Paper Name: Safe Exploration by Solving Early Terminated MDP
        Paper author: Hao Sun, Ziping Xu, Meng Fang, Zhenghao Peng, Jiadong Guo, Bo Dai, Bolei Zhou
        Paper URL: https://arxiv.org/abs/2107.04200
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        env_id,
        cfgs,
        algo='ppo_lag_early_terminated',
        wrapper_type: str = 'EarlyTerminatedEnvWrapper',
    ) -> None:
        r"""Initialize PPO_Lag_Earyly_Terminated."""
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
            algo=algo,
            wrapper_type=wrapper_type,
        )
