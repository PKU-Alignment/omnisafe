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

import helpers
import omnisafe


@helpers.parametrize(
    algo=[
        'PolicyGradient',
        'PPO',
        'PPOLag',
        'NaturalPG',
        'TRPO',
        'TRPOLag',
        'PDO',
        'NPGLag',
        'CPO',
        'PCPO',
        'FOCOPS',
        'CPPOPid',
        'DDPG',
    ]
)
def test_on_policy(algo):
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {'epochs': 1, 'steps_per_epoch': 1000, 'pi_iters': 1, 'critic_iters': 1}
    env = omnisafe.Env(env_id)
    agent = omnisafe.Agent(algo, env, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()
