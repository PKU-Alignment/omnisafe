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
        'actor_iters': 1,
        'critic_iters': 1,
    }
    agent = omnisafe.Agent(on_policy_algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()


@helpers.parametrize(off_policy_algo=omnisafe.ALGORITHMS['off-policy'])
def test_off_policy(off_policy_algo):
    """Test algorithms"""
    env_id = 'SafetyPointGoal1-v0'
    custom_cfgs = {'epochs': 1, 'steps_per_epoch': 2000}
    agent = omnisafe.Agent(off_policy_algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    agent.learn()


def test_evaluate_saved_policy():
    """Test render policy."""
    DIR = os.path.join(os.path.dirname(__file__), 'runs')
    evaluator = omnisafe.Evaluator()
    for env in os.scandir(DIR):
        env_path = os.path.join(DIR, env)
        for algo in os.scandir(env_path):
            print(algo)
            algo_path = os.path.join(env_path, algo)
            for exp in os.scandir(algo_path):
                exp_path = os.path.join(algo_path, exp)
                for item in os.scandir(os.path.join(exp_path, 'torch_save')):
                    if item.is_file() and item.name.split('.')[-1] == 'pt':
                        evaluator.load_saved_model(save_dir=exp_path, model_name=item.name)
                        evaluator.evaluate(num_episodes=1)
                        evaluator.render(num_episode=1, camera_name='track', width=256, height=256)
