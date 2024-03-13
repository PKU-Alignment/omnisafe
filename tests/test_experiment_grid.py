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
"""Test experiment grid."""

import os

from omnisafe.common.experiment_grid import ExperimentGrid
from omnisafe.utils.exp_grid_tools import train


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
    eg.add('algo_cfgs:steps_per_epoch', [1024])
    eg.add('model_cfgs:actor:activation', ['tanh'])
    eg.add('train_cfgs:total_steps', [1024])
    eg.add('seed', [0])
    # total experiment num must can be divided by num_pool
    # meanwhile, users should decide this value according to their machine
    eg.run(train, num_pool=1)

    eg.analyze('algo')
    # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
    eg.evaluate(num_episodes=1)


# def test_modelbased_experiment_grid():
#     """Test experiment grid."""
#     eg = ExperimentGrid(exp_name='Test_experiment_grid_modelbased')

#     # Set the environments.
#     mujoco_envs = ['SafetyPointGoal1-v0-modelbased']

#     # Set the algorithms.
#     eg.add('env_id', mujoco_envs)

#     eg.add('algo', ['LOOP'])
#     eg.add('train_cfgs:total_steps', [1024])
#     eg.add('train_cfgs:vector_env_nums', [1])
#     eg.add('train_cfgs:torch_threads', [1])
#     eg.add('algo_cfgs:obs_normalize', [False])
#     eg.add('algo_cfgs:steps_per_epoch', [1000])
#     eg.add('algo_cfgs:action_repeat', [10])
#     eg.add('algo_cfgs:update_dynamics_cycle', [2000])
#     eg.add('algo_cfgs:update_policy_cycle', [2000])
#     eg.add('algo_cfgs:update_policy_iters', [1])
#     eg.add('algo_cfgs:start_learning_steps', [3])
#     eg.add('algo_cfgs:policy_batch_size', [1])
#     eg.add('dynamics_cfgs:num_ensemble', [5])
#     eg.add('dynamics_cfgs:batch_size', [1])
#     eg.add('dynamics_cfgs:max_epoch', [1])
#     eg.add('dynamics_cfgs:predict_cost', [False])
#     eg.add('planner_cfgs:plan_horizon', [2])
#     eg.add('planner_cfgs:num_particles', [5])
#     eg.add('planner_cfgs:num_samples', [10])
#     eg.add('planner_cfgs:num_elites', [5])
#     eg.add('evaluation_cfgs:use_eval', [False])
#     eg.add('logger_cfgs:use_wandb', [False])
#     eg.add('seed', [0])
#     # total experiment num must can be divided by num_pool
#     # meanwhile, users should decide this value according to their machine
#     eg.run(train, num_pool=1)

#     eg.analyze('algo')
#     # eg.render(num_episodes=1, render_mode='rgb_array', width=256, height=256)
#     eg.evaluate(num_episodes=1)


def teardown_module():
    """teardown_module."""
    base_path = os.path.dirname(os.path.abspath(__file__))

    # remove exp-x folder
    exp_x_path = os.path.join(base_path, 'exp-x')
    if os.path.exists(exp_x_path):
        os.system(f'rm -rf {exp_x_path}')

    # remove png
    os.system(f'rm -rf {base_path}/algo--*.png')
