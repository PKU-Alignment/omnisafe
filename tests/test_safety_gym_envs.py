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
"""Test Environments"""

import helpers
import omnisafe


@helpers.parametrize(
    algo=['PPOSimmerQ', 'PPOLag', 'PPOSaute', 'PPOEarlyTerminated'],
    agent_id=['Point', 'Car', 'Racecar'],
    env_id=[
        'Goal',
        'Button',
        'Push',
    ],
    level=['1'],
)
def test_safety_nvigation(algo, agent_id, env_id, level):
    """Test environments"""
    env_id = 'Safety' + agent_id + env_id + level + '-v0'
    # env_id = 'PointGoal1'
    custom_cfgs = {
        'epochs': 1,
        'steps_per_epoch': 1000,
        'pi_iters': 1,
        'critic_iters': 1,
        'env_cfgs': {'num_envs': 1},
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    # agent.set_seed(seed=0)
    agent.learn()


@helpers.parametrize(
    algo=['PPOSimmerQ', 'PPOLag', 'PPOSaute', 'PPOEarlyTerminated'],
    agent_id=['Ant', 'Humanoid', 'Walker2d', 'Hopper', 'HalfCheetah', 'Swimmer'],
    env_id=['Velocity'],
)
def test_safety_velocity(algo, agent_id, env_id):
    """Test environments"""
    env_id = 'Safety' + agent_id + env_id + '-v4'
    # env_id = 'PointGoal1'
    custom_cfgs = {
        'epochs': 1,
        'steps_per_epoch': 1000,
        'pi_iters': 1,
        'critic_iters': 1,
        'env_cfgs': {'num_envs': 1},
        'parallel': 1,
    }
    agent = omnisafe.Agent(algo, env_id, custom_cfgs=custom_cfgs, parallel=1)
    # agent.set_seed(seed=0)
    agent.learn()
