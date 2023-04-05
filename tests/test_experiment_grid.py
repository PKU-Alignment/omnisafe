# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Test experiment grid"""

import os
import sys

import omnisafe
from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.typing import Tuple


def train(
    exp_id: str,
    algo: str,
    env_id: str,
    custom_cfgs: dict,
) -> Tuple[float, float, float]:
    """Train a policy from exp-x config with OmniSafe.

    Args:
        exp_id (str): Experiment ID.
        algo (str): Algorithm to train.
        env_id (str): The name of test environment.
        custom_cfgs (dict): Custom configurations.
        num_threads (int, optional): Number of threads. Defaults to 6.
    """
    terminal_log_name = 'terminal.log'
    error_log_name = 'error.log'
    if 'seed' in custom_cfgs:
        terminal_log_name = f'seed{custom_cfgs["seed"]}_{terminal_log_name}'
        error_log_name = f'seed{custom_cfgs["seed"]}_{error_log_name}'
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    print(f'exp-x: {exp_id} is training...')
    if not os.path.exists(custom_cfgs['logger_cfgs']['log_dir']):
        os.makedirs(custom_cfgs['logger_cfgs']['log_dir'], exist_ok=True)
    # pylint: disable-next=consider-using-with
    sys.stdout = open(  # noqa: SIM115
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', terminal_log_name),
        'w',
        encoding='utf-8',
    )
    # pylint: disable-next=consider-using-with
    sys.stderr = open(  # noqa: SIM115
        os.path.join(f'{custom_cfgs["logger_cfgs"]["log_dir"]}', error_log_name),
        'w',
        encoding='utf-8',
    )
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    reward, cost, ep_len = agent.learn()
    return reward, cost, ep_len


def test_experiment_grid():
    """Test experiment grid."""
    eg = ExperimentGrid(exp_name='Test_experiment_grid')

    # Set the environments.
    mujoco_envs = ['SafetyAntVelocity-v1']

    # Set the algorithms.
    eg.add('env_id', mujoco_envs)

    eg.add('algo', ['PPO'])
    eg.add('logger_cfgs:use_wandb', [False])
    eg.add('train_cfgs:vector_env_nums', [1])
    eg.add('train_cfgs:torch_threads', [1])
    eg.add('algo_cfgs:update_cycle', [1024])
    eg.add('train_cfgs:total_steps', [1024])
    eg.add('seed', [0])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1)

    eg.analyze('algo')
    # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)
