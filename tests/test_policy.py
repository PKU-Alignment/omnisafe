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
"""Test policy algorithms."""

import os

import pytest
import torch

import helpers
import omnisafe
import simple_env  # noqa: F401


base_policy = ['PolicyGradient', 'NaturalPG', 'TRPO', 'PPO']
naive_lagrange_policy = ['PPOLag', 'TRPOLag', 'RCPO', 'OnCRPO', 'PDO']
first_order_policy = ['CUP', 'FOCOPS']
second_order_policy = ['CPO', 'PCPO']
penalty_policy = ['P3O', 'IPO']
off_base = ['DDPG', 'TD3']
off_lag = ['DDPGLag', 'TD3Lag', 'DDPGPID', 'TD3PID']
sac_base = ['SAC']
sac_lag = ['SACLag', 'SACPID']
saute_policy = ['TRPOSaute', 'PPOSaute']
simmer_policy = ['TRPOSimmerPID', 'PPOSimmerPID']
pid_lagrange_policy = ['TRPOPID', 'CPPOPID']
early_terminated_policy = ['TRPOEarlyTerminated', 'PPOEarlyTerminated']
offline_policy = ['BCQ', 'BCQLag', 'CRR', 'CCRR', 'VAEBC']

model_cfgs = {
    'linear_lr_decay': True,
    'actor': {
        'hidden_sizes': [8, 8],
    },
    'critic': {
        'hidden_sizes': [8, 8],
    },
}

optim_case = [0, 1, 2, 3, 4]


@helpers.parametrize(optim_case=optim_case)
def test_cpo(optim_case):
    agent = omnisafe.Agent('CPO', 'Simple-v0', custom_cfgs={})
    b_grads = torch.Tensor([1])
    ep_costs = torch.Tensor([-1])
    r = torch.Tensor([0])
    q = torch.Tensor([0])
    s = torch.Tensor([1])
    p = torch.Tensor([1])
    xHx = torch.Tensor([1])
    x = torch.Tensor([1])
    A = torch.Tensor([1])
    B = torch.Tensor([1])
    assert agent.agent._determine_case(b_grads, ep_costs, q, r, s)[0] == 3
    s = torch.Tensor([-1])
    assert agent.agent._determine_case(b_grads, ep_costs, q, r, s)[0] == 2
    ep_costs = torch.Tensor([1])
    assert agent.agent._determine_case(b_grads, ep_costs, q, r, s)[0] == 1
    s = torch.Tensor([1])
    assert agent.agent._determine_case(b_grads, ep_costs, q, r, s)[0] == 0
    step_direction, lambda_star, nu_star = agent.agent._step_direction(
        optim_case=optim_case,
        xHx=xHx,
        x=x,
        A=A,
        B=B,
        q=q,
        p=p,
        r=r,
        s=s,
        ep_costs=ep_costs,
    )
    assert isinstance(step_direction, torch.Tensor)
    assert isinstance(lambda_star, torch.Tensor)
    assert isinstance(nu_star, torch.Tensor)
    step_direction = torch.as_tensor(1000000.0).unsqueeze(0)
    grads = torch.as_tensor(torch.inf).unsqueeze(0)
    p_dist = torch.distributions.Normal(torch.Tensor([0.0]), torch.Tensor([1.0]))
    obs = torch.Tensor([1.0, 1.0, 1.0])
    act = torch.Tensor([1.0, 1.0])
    logp = torch.Tensor([1.0])
    adv_r = torch.Tensor([1.0])
    adv_c = torch.Tensor([1.0])
    loss_reward_before = torch.Tensor([1.0])
    loss_cost_before = torch.Tensor([1.0])
    step_direction, acceptance_step = agent.agent._cpo_search_step(
        step_direction=step_direction,
        grads=grads,
        p_dist=p_dist,
        obs=obs,
        act=act,
        logp=logp,
        adv_r=adv_r,
        adv_c=adv_c,
        loss_reward_before=loss_reward_before,
        loss_cost_before=loss_cost_before,
    )


def test_assertion_error():
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'use_tensorboard': True,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
    }
    with pytest.raises(AssertionError):
        agent = omnisafe.Agent('NotExist', env_id, custom_cfgs=custom_cfgs)
    with pytest.raises(AssertionError):
        custom_cfgs['train_cfgs']['vector_env_nums'] = 2
        agent = omnisafe.Agent('PPOEarlyTerminated', env_id, custom_cfgs=custom_cfgs)
        agent.learn()
    with pytest.raises(AssertionError):
        custom_cfgs['train_cfgs']['vector_env_nums'] = 2
        agent = omnisafe.Agent('TRPOEarlyTerminated', env_id, custom_cfgs=custom_cfgs)
        agent.learn()
    custom_cfgs['train_cfgs']['vector_env_nums'] = 1
    with pytest.raises(AssertionError):
        agent = omnisafe.Agent(111, env_id, custom_cfgs=custom_cfgs)
    with pytest.raises(AssertionError):
        agent = omnisafe.Agent('PPO', 'NotExist', custom_cfgs=custom_cfgs)
    with pytest.raises(AssertionError):
        custom_cfgs['train_cfgs']['parallel'] = 2
        agent = omnisafe.Agent('DDPG', env_id, custom_cfgs=custom_cfgs)
        agent.learn()
    with pytest.raises(AssertionError):
        custom_cfgs['train_cfgs']['parallel'] = 'abc'
        agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    with pytest.raises(AssertionError):
        custom_cfgs['train_cfgs']['parallel'] = 0
        agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    with pytest.raises(AssertionError):
        custom_cfgs = [1, 2, 3]
        agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)


def test_render():
    """Test render image"""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
    }
    agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    agent.render(num_episodes=1, render_mode='rgb_array')


@helpers.parametrize(algo=['PETS', 'CCEPETS', 'CAPPETS', 'RCEPETS'])
def test_cem_based(algo):
    """Test model_based algorithms."""
    env_id = 'Simple-v0'

    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'obs_normalize': True,
            'steps_per_epoch': 100,
            'action_repeat': 1,
            'update_dynamics_cycle': 100,
            'start_learning_steps': 3,
        },
        'dynamics_cfgs': {
            'num_ensemble': 5,
            'batch_size': 10,
            'max_epoch': 1,
            'predict_cost': True,
        },
        'planner_cfgs': {
            'plan_horizon': 2,
            'num_particles': 5,
            'num_samples': 10,
            'num_elites': 5,
        },
        'evaluation_cfgs': {
            'use_eval': True,
            'eval_cycle': 100,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()
    agent.render()


@helpers.parametrize(algo=['LOOP', 'SafeLOOP'])
def test_loop(algo):
    """Test model_based algorithms."""
    env_id = 'Simple-v0'

    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'obs_normalize': True,
            'use_cost': True,
            'steps_per_epoch': 100,
            'action_repeat': 1,
            'update_dynamics_cycle': 100,
            'update_policy_cycle': 100,
            'update_policy_iters': 1,
            'start_learning_steps': 3,
            'policy_batch_size': 10,
            'use_critic_norm': True,
            'max_grad_norm': True,
        },
        'planner_cfgs': {
            'plan_horizon': 2,
            'num_particles': 5,
            'num_samples': 10,
            'num_elites': 5,
        },
        'dynamics_cfgs': {
            'num_ensemble': 5,
            'batch_size': 10,
            'max_epoch': 1,
            'predict_cost': True,
        },
        'evaluation_cfgs': {
            'use_eval': True,
            'eval_cycle': 100,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(algo=off_base)
def test_off_policy(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_cycle': 50,
            'update_iters': 2,
            'start_learning_steps': 0,
            'use_critic_norm': True,
            'max_grad_norm': True,
            'obs_normalize': True,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(algo=off_lag)
def test_off_lag_policy(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_cycle': 50,
            'update_iters': 2,
            'start_learning_steps': 0,
            'use_critic_norm': True,
            'max_grad_norm': True,
            'obs_normalize': True,
            'warmup_epochs': 0,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


auto_alpha = [True, False]


@helpers.parametrize(auto_alpha=auto_alpha)
def test_sac_policy(auto_alpha):
    """Test sac algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_cycle': 50,
            'update_iters': 2,
            'start_learning_steps': 0,
            'auto_alpha': auto_alpha,
            'use_critic_norm': True,
            'max_grad_norm': True,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent('SAC', env_id, custom_cfgs=custom_cfgs)
    agent.learn()


auto_alpha = [True, False]


@helpers.parametrize(auto_alpha=auto_alpha, algo=sac_lag)
def test_sac_lag_policy(auto_alpha, algo):
    """Test sac algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_cycle': 50,
            'update_iters': 2,
            'start_learning_steps': 0,
            'auto_alpha': auto_alpha,
            'use_critic_norm': True,
            'max_grad_norm': True,
            'warmup_epochs': 0,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
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
        + early_terminated_policy
    ),
)
def test_on_policy(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(algo=pid_lagrange_policy)
def test_pid(algo):
    """Test pid algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'lagrange_cfgs': {
            'diff_norm': True,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(
    algo=(offline_policy),
)
def test_offline(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    dataset = os.path.join(os.path.dirname(__file__), 'saved_source', 'Simple-v0.npz')
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 4,
            'torch_threads': 4,
            'dataset': dataset,
        },
        'algo_cfgs': {
            'batch_size': 10,
            'steps_per_epoch': 2,
        },
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(
    fn_type=['softchi', 'chisquare'],
)
def test_coptidice(fn_type):
    """Test coptidice algorithms."""
    env_id = 'Simple-v0'
    dataset = os.path.join(os.path.dirname(__file__), 'saved_source', 'Simple-v0.npz')
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 4,
            'torch_threads': 4,
            'dataset': dataset,
        },
        'algo_cfgs': {
            'batch_size': 10,
            'steps_per_epoch': 2,
            'fn_type': fn_type,
        },
    }
    agent = omnisafe.Agent('COptiDICE', env_id, custom_cfgs=custom_cfgs)
    agent.learn()


@helpers.parametrize(algo=['PPO', 'SAC', 'PPOLag'])
def test_workflow_for_training(algo):
    """Test base algorithms."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': model_cfgs,
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs)
    agent.learn()

    agent.plot(smooth=2)
    # agent.render(num_episodes=1, render_mode='rgb_array', width=1, height=1)
    agent.evaluate(num_episodes=1)


def test_std_anealing():
    """Test std_anealing."""
    env_id = 'Simple-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 200,
            'vector_env_nums': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 100,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
            'save_model_freq': 1,
        },
        'model_cfgs': {
            'actor': {
                'hidden_sizes': [8, 8],
            },
            'critic': {
                'hidden_sizes': [8, 8],
            },
            'exploration_noise_anneal': True,
        },
    }
    agent = omnisafe.Agent('PPO', env_id, custom_cfgs=custom_cfgs)
    agent.learn()


def teardown_module():
    """teardown_module."""
    # remove runs folder
    current_path = os.path.dirname(os.path.abspath(__file__))
    runs_path = os.path.join(current_path, 'runs')
    if os.path.exists(runs_path):
        os.system(f'rm -rf {runs_path}')
