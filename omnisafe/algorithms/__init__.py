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
"""Safe Reinforcement Learning algorithms."""

# Off Policy Safe
from omnisafe.algorithms.off_policy.ddpg import DDPG

# On Policy Safe
from omnisafe.algorithms.on_policy.cpo import CPO
from omnisafe.algorithms.on_policy.cppo_pid import CPPOPid
from omnisafe.algorithms.on_policy.cup import CUP
from omnisafe.algorithms.on_policy.focops import FOCOPS
from omnisafe.algorithms.on_policy.natural_pg import NaturalPG
from omnisafe.algorithms.on_policy.npg_lag import NPGLag
from omnisafe.algorithms.on_policy.pcpo import PCPO
from omnisafe.algorithms.on_policy.pdo import PDO
from omnisafe.algorithms.on_policy.policy_gradient import PolicyGradient
from omnisafe.algorithms.on_policy.ppo import PPO
from omnisafe.algorithms.on_policy.ppo_early_terminated import PPOEarlyTerminated
from omnisafe.algorithms.on_policy.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.ppo_saute import PPOSaute
from omnisafe.algorithms.on_policy.ppo_simmer_pid import PPOSimmerPID
from omnisafe.algorithms.on_policy.ppo_simmer_q import PPOSimmerQ
from omnisafe.algorithms.on_policy.trpo import TRPO
from omnisafe.algorithms.on_policy.trpo_lag import TRPOLag


algo_type = {
    'off-policy': ['DDPG'],
    'on-policy': [
        'CPO',
        'FOCOPS',
        'CPPOPid',
        'FOCOPS',
        'NaturalPG',
        'NPGLag',
        'PCPO',
        'PDO',
        'PolicyGradient',
        'PPO',
        'PPOLag',
        'TRPO',
        'TRPOLag',
        'CUP',
        'PPOSaute',
        'PPOSimmerPID',
        'PPOSimmerQ',
        'PPOEarlyTerminated',
    ],
    'model-based': ['MBPPOLag', 'SafeLoop'],
}
