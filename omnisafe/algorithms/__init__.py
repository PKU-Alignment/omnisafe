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

from omnisafe.algorithms.off_policy.ddpg import DDPG
from omnisafe.algorithms.off_policy.ddpg_lag import DDPGLag
from omnisafe.algorithms.off_policy.sac import SAC
from omnisafe.algorithms.off_policy.sac_lag import SACLag
from omnisafe.algorithms.off_policy.sddpg import SDDPG
from omnisafe.algorithms.off_policy.td3 import TD3
from omnisafe.algorithms.off_policy.td3_lag import TD3Lag
from omnisafe.algorithms.on_policy.base.natural_pg import NaturalPG
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.algorithms.on_policy.base.trpo import TRPO
from omnisafe.algorithms.on_policy.early_terminated.ppo_early_terminated import PPOEarlyTerminated
from omnisafe.algorithms.on_policy.early_terminated.ppo_lag_early_terminated import (
    PPOLagEarlyTerminated,
)
from omnisafe.algorithms.on_policy.first_order.cup import CUP
from omnisafe.algorithms.on_policy.first_order.focops import FOCOPS
from omnisafe.algorithms.on_policy.naive_lagrange.npg_lag import NPGLag
from omnisafe.algorithms.on_policy.naive_lagrange.pdo import PDO
from omnisafe.algorithms.on_policy.naive_lagrange.ppo_lag import PPOLag
from omnisafe.algorithms.on_policy.naive_lagrange.trpo_lag import TRPOLag
from omnisafe.algorithms.on_policy.pid_lagrange.cppo_pid import CPPOPid
from omnisafe.algorithms.on_policy.pid_lagrange.trpo_pid import TRPOPid
from omnisafe.algorithms.on_policy.saute.ppo_lag_saute import PPOLagSaute
from omnisafe.algorithms.on_policy.saute.ppo_saute import PPOSaute
from omnisafe.algorithms.on_policy.second_order.cpo import CPO
from omnisafe.algorithms.on_policy.second_order.pcpo import PCPO
from omnisafe.algorithms.on_policy.simmer.ppo_lag_simmer_pid import PPOLagSimmerPid
from omnisafe.algorithms.on_policy.simmer.ppo_lag_simmer_q import PPOLagSimmerQ
from omnisafe.algorithms.on_policy.simmer.ppo_simmer_pid import PPOSimmerPid
from omnisafe.algorithms.on_policy.simmer.ppo_simmer_q import PPOSimmerQ


algo_type = {
    'off-policy': [
        'DDPG',
        'DDPGLag',
        'TD3',
        'TD3Lag',
        'SAC',
        'SACLag',
        'SDDPG',
    ],
    'on-policy': [
        'PolicyGradient',
        'NaturalPG',
        'TRPO',
        'PPO',
        'PDO',
        'NPGLag',
        'TRPOLag',
        'PPOLag',
        'CPPOPid',
        'TRPOPid',
        'FOCOPS',
        'CUP',
        'CPO',
        'PCPO',
        'PPOSimmerPid',
        'PPOSimmerQ',
        'PPOLagSimmerQ',
        'PPOLagSimmerPid',
        'PPOSaute',
        'PPOLagSaute',
        'PPOEarlyTerminated',
        'PPOLagEarlyTerminated',
    ],
    'model-based': ['MBPPOLag', 'SafeLoop'],
}
