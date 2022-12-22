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
"""Environment wrappers."""

import itertools
from types import MappingProxyType

from omnisafe.wrappers.early_terminated_wrapper import EarlyTerminatedEnvWrapper
from omnisafe.wrappers.off_policy_wrapper import OffPolicyEnvWrapper
from omnisafe.wrappers.on_policy_wrapper import OnPolicyEnvWrapper
from omnisafe.wrappers.saute_wrapper import SauteEnvWrapper
from omnisafe.wrappers.simmer_wrapper import SimmerEnvWrapper


ENVWRAPPERS = {
    'on-policy-wrapper': OnPolicyEnvWrapper,
    'off-policy-wrapper': OffPolicyEnvWrapper,
    'saute-wrapper': SauteEnvWrapper,
    'simmer-wrapper': SimmerEnvWrapper,
    'early-terminated-wrapper': EarlyTerminatedEnvWrapper,
}

ENVWRAPPERS2TYPE = {
    env_wrapper: env_wrapper_type for env_wrapper_type, env_wrapper in ENVWRAPPERS.items()
}

__all__ = ENVWRAPPERS['all'] = tuple(itertools.chain(ENVWRAPPERS.values()))

assert len(ENVWRAPPERS2TYPE) == len(__all__), 'Duplicate environment wrappers found.'

ENVWRAPPERS = MappingProxyType(ENVWRAPPERS)
ENVWRAPPERS2TYPE = MappingProxyType(ENVWRAPPERS2TYPE)

del itertools, MappingProxyType
