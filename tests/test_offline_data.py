# # Copyright 2023 OmniSafe Team. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """Test offline module."""

# import os

# from omnisafe.common.offline.data_collector import OfflineDataCollector


# def test_data_collector():
#     env_name = 'SafetyPointGoal1-v0'
#     size = 2_000
#     base_dir = os.path.dirname(__file__)
#     agents = [
#         (
#             os.path.join(
#                 base_dir,
#                 'saved_source',
#                 'PPO-{SafetyPointGoal1-v0}',
#                 'seed-000-2023-03-16-12-08-52',
#             ),
#             'epoch-0.pt',
#             2_000,
#         ),
#     ]
#     save_dir = os.path.join(base_dir, 'saved_data')

#     col = OfflineDataCollector(size, env_name)
#     for agent, model_name, num in agents:
#         col.register_agent(agent, model_name, num)
#     col.collect(save_dir)

#     # delete the saved data
#     os.system(f'rm -rf {save_dir}')
