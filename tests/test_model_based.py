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
        'MBPPOLag',
        'SafeLoop',
    ],
    env_id=[
        'SafetyPointGoal1-v0',
        'SafetyPointGoal3-v0',
        'SafetyCarGoal1-v0',
        'SafetyCarGoal3-v0',
    ],
    device=[
        'cpu',
    ],
)
def test_model_based(algo, env_id, device):
    seed = 0
    custom_cfgs = {
        'max_real_time_steps': 10000,
        'pi_iters': 1,
        'critic_iters': 1,
        'imaging_steps_per_policy_update': 30000,
        'mixed_real_time_steps': 1500,
        'update_dynamics_freq': 10000,
        'update_policy_freq': 10000,
        'update_policy_start_timesteps': 0,
        'update_policy_iters': 1,
        'log_freq': 10000,
    }
    env = omnisafe.EnvModelBased(algo, env_id)

    agent = omnisafe.Agent(algo, env, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()


if __name__ == '__main__':
    test_model_based(algo='MBPPOLag', env_id='SafetyPointGoal1-v0', devide='cpu')
