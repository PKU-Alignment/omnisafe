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
"""Example of training a policy with OmniSafe."""

import omnisafe
import simple_env  # noqa: F401


if __name__ == '__main__':
    algo = 'NaturalPG'
    env_id = 'Test-v0'
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 4096,
            'vector_env_nums': 1,
        },
        'algo_cfgs': {
            'steps_per_epoch': 1024,
            'update_iters': 2,
        },
        'logger_cfgs': {
            'use_wandb': False,
        },
    }
    train_terminal_cfgs = {
        'parallel': 2,
    }
    agent = omnisafe.Agent(algo, env_id, train_terminal_cfgs, custom_cfgs)
    agent.learn()
