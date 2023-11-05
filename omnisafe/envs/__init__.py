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
"""Environment API for OmniSafe."""

import itertools
from types import MappingProxyType

from omnisafe.envs.core import CMDP, env_register, make, support_envs
from omnisafe.envs.discrete_env import DiscreteEnv
from omnisafe.envs.mujoco_env import MujocoEnv
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.envs.safety_gymnasium_modelbased import SafetyGymnasiumModelBased


ENVIRONMENTS = {
    'box': tuple(
        MujocoEnv.support_envs()
        + SafetyGymnasiumEnv.support_envs()
        + SafetyGymnasiumModelBased.support_envs(),
    ),
    'discrete': tuple(DiscreteEnv.support_envs()),
}

ENVIRONMNET2TYPE = {
    env: env_type for env_type, environments in ENVIRONMENTS.items() for env in environments
}

__all__ = ENVIRONMENTS['all'] = tuple(itertools.chain.from_iterable(ENVIRONMENTS.values()))

assert len(ENVIRONMNET2TYPE) == len(__all__), 'Duplicate algorithm names found.'

ENVIRONMENTS = MappingProxyType(ENVIRONMENTS)  # make this immutable
ENVIRONMNET2TYPE = MappingProxyType(ENVIRONMNET2TYPE)  # make this immutable

del itertools, MappingProxyType
