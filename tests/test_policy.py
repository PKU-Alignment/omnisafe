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
"""Test policy algorithms"""

import os

import helpers
import omnisafe


@helpers.parametrize(on_policy_algo=omnisafe.ALGORITHMS['on-policy'])
def test_on_policy(on_policy_algo):
    """Test algorithms"""
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {
        'epochs': 1,
        'steps_per_epoch': 2000,
        'pi_iters': 1,
        'critic_iters': 1,
        'env_cfgs': {'num_envs': 1},
    }
    agent = omnisafe.Agent(on_policy_algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()


@helpers.parametrize(off_policy_algo=omnisafe.ALGORITHMS['off-policy'])
def test_off_policy(off_policy_algo):
    """Test algorithms"""
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {'epochs': 1, 'steps_per_epoch': 2000, 'pi_iters': 1, 'critic_iters': 1}
    agent = omnisafe.Agent(off_policy_algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()
