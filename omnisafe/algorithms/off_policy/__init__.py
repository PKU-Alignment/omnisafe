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
"""Off-policy algorithms."""

from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.algorithms.off_policy.ddpg_lag import DDPGLag
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.algorithms.off_policy.td3_lag import TD3Lag


__all__ = ['DDPG', 'TD3', 'SAC', 'DDPGLag', 'TD3Lag', 'SACLag']
