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
"""Example of training a policy from exp-x config with OmniSafe."""

import os
import sys

import torch

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid


def train(exp_id, algo, env_id, custom_cfgs):
    """Train a policy from exp-x config with OmniSafe."""
    torch.set_num_threads(6)
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    USE_REDIRECTION = True
    if USE_REDIRECTION:
        if not os.path.exists(custom_cfgs['data_dir']):
            os.makedirs(custom_cfgs['data_dir'])
        sys.stdout = open(f'{custom_cfgs["data_dir"]}terminal.log', 'w', encoding='utf-8')
        sys.stderr = open(f'{custom_cfgs["data_dir"]}error.log', 'w', encoding='utf-8')
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


if __name__ == '__main__':
    eg = ExperimentGrid(exp_name='Safety_Gymnasium_Goal')
    eg.add('algo', ['PPO', 'PPOLag'])
    eg.add('env_id', ['SafetyPointGoal1-v0'])
    eg.add('epochs', 1)
    eg.add('actor_lr', [0.001, 0.003, 0.004], 'lr', True)
    eg.add('actor_iters', [1, 2], 'ac_iters', True)
    eg.add('seed', [0, 5, 10])
    # eg.add('env_cfgs:num_envs', [2, 3])
    eg.run(train, num_pool=5)
