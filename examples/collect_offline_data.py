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

ENV_NAME = 'SafetyPointCircle1-v0'
SIZE = 2_000_000
AGENTS = [
    ('./runs/PPO', 'epoch-500', 1_000_000),
    ('./runs/PPOLag', 'epoch-500', 1_000_000),
]
SAVE_DIR = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(SIZE, ENV_NAME)
    for agent, model_name, num in AGENTS:
        col.register_agent(agent, model_name, num)
    col.collect(SAVE_DIR)
