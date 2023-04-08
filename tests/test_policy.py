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
"""Test policy algorithms"""

import helpers
import omnisafe
import simple_env  # noqa: F401


base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
first_order_policy = ['CUP', 'FOCOPS']
second_order_policy = ['CPO', 'PCPO']
penalty_policy = ['P3O', 'IPO']
off_policy = ['DDPG', 'TD3', 'SAC', 'DDPGLag', 'TD3Lag', 'SACLag']
saute_policy = ['TRPOSaute']
simmer_policy = ['TRPOSimmerPID']
# pid_lagrange_policy = ['CPPOPid', 'TRPOPid']
# early_terminated_policy = ['PPOEarlyTerminated', 'PPOLagEarlyTerminated']
# saute_policy = ['PPOSaute', 'PPOLagSaute']
# simmer_policy = ['PPOSimmerQ', 'PPOLagSimmerQ', 'PPOSimmerPid', 'PPOLagSimmerPid']
# model_based_policy = ['MBPPOLag', 'SafeLOOP', 'CAP']


@helpers.parametrize(algo=off_policy)
def test_off_policy(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 2048,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'update_cycle': 1024,
            'steps_per_sample': 1024,
            'update_iters': 1,
            'start_learning_steps': 0,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(
    algo=(
        base_policy
        + naive_lagrange_policy
        + first_order_policy
        + second_order_policy
        + penalty_policy
        + saute_policy
        + simmer_policy
    ),
)
def test_on_policy(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 2048,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'update_cycle': 1024,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(algo=['PPO', 'SAC', 'PPOLag'])
def test_workflow_for_training(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 2048,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'update_cycle': 1024,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=1)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=1, height=1)
    agent.evaluate(num_episodes=1)


def test_std_anealing():
    """Test std_anealing."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 2048,
            'vector_env_nums': 1,
        },
        'algo_cfgs': {
            'update_cycle': 1024,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': {
            'exploration_noise_anneal': True,
        },
    }
    agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()


# @helpers.parametrize(algo=['PPOLag'])
# def test_cuda(algo):
#    """Test std_anealing."""
#    env_id = 'Simple-v0'
#    custom_cfgs = {
#        'train_cfgs': {
#            'total_steps': 2048,
#            'vector_env_nums': 1,
#            'torch_threads': 4,
#            'device': 'cuda:0',
#        },
#        'algo_cfgs': {
#            'update_cycle': 1024,
#            'update_iters': 2,
#        },
#        'logger_cfgs': {
#            'use_wandb': False,
#            'save_model_freq': 1,
#        },
#    }
#    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
#    agent.learn()


# @helpers.parametrize(off_policy_algo=omnisafe.ALGORITHMS['off-policy'])
# def test_off_policy(off_policy_algo):
#     """Test off policy algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'update_after': 999,
#         'update_every': 1,
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(off_policy_algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=naive_lagrange_policy)
# def test_naive_lagrange_policy(algo):
#     """Test naive lagrange algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=first_order_policy)
# def test_first_order_policy(algo):
#     """Test first order algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=second_order_policy)
# def test_second_order_policy(algo):
#     """Test second order algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'cost_limit': 0.01,
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=pid_lagrange_policy)
# def test_pid_lagrange_policy(algo):
#     """Test pid lagrange algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=penalty_policy)
# def test_penalty_policy(algo):
#     """Test penalty algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'parallel': 2,
#         'cost_limit': 0.01,
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=early_terminated_policy)
# def test_early_terminated_policy(algo):
#     """Test early terminated algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=saute_policy)
# def test_saute_policy(algo):
#     """Test Saute algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# @helpers.parametrize(algo=simmer_policy)
# def test_simmer_policy(algo):
#     """Test Simmer algorithms."""
#     env_id = 'SafetyHumanoidVelocity-v1'
#     custom_cfgs = {
#         'epochs': 1,
#         'steps_per_epoch': 1000,
#         'pi_iters': 1,
#         'critic_iters': 1,
#         'env_cfgs': {'num_envs': 1},
#         'use_wandb': False,
#     }
#     agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
#     agent.learn()


# def test_evaluate_saved_policy():
#     """Test evaluate policy."""
#     DIR = os.path.join(os.path.dirname(__file__), 'saved_policy')
#     evaluator = omnisafe.Evaluator()
#     for algo in os.scandir(DIR):
#         algo_path = os.path.join(DIR, algo)
#         for exp in os.scandir(algo_path):
#             exp_path = os.path.join(algo_path, exp)
#             for item in os.scandir(os.path.join(exp_path, 'torch_save')):
#                 if item.is_file() and item.name.split('.')[-1] == 'pt':
#                     evaluator.load_saved_model(save_dir=exp_path, model_name=item.name)
#                     evaluator.evaluate(num_episodes=1)
#                     evaluator.render(num_episode=1, camera_name='track', width=256, height=256)
