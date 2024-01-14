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
"""Example of collecting offline data with OmniSafe."""

from omnisafe.common.offline.data_collector import OfflineDataCollector


# please change agent path and env name
# also, please make sure you have run:
# python train_policy.py --algo PPO --env ENVID
# where ENVID is the environment from which you want to collect data.
# The `PATH_TO_AGENT` is the directory path containing the `torch_save`.

env_name = 'SafetyAntVelocity-v1'
size = 1_000_000
agents = [
    ('PATH_TO_AGENT', 'epoch-500.pt', 1_000_000),
]
save_dir = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)
