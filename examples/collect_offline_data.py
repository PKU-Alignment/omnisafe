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
env_name = 'SafetyPointCircle1-v0'
size = 2_000_000
agents = [
    ('./runs/PPO', 'epoch-500', 1_000_000),
    ('./runs/PPOLag', 'epoch-500', 1_000_000),
]
save_dir = './data'

if __name__ == '__main__':
    col = OfflineDataCollector(size, env_name)
    for agent, model_name, num in agents:
        col.register_agent(agent, model_name, num)
    col.collect(save_dir)
