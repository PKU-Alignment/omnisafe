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
# This file is just to get around a baselines import hack.
# env_type is set based on the final part of the entry_point module name.
# In the regular gym mujoco envs this is 'mujoco'.
# We want baselines to treat these as mujoco envs, so we redirect from here,
# and ensure the registry entries are pointing at this file as well.

from safety_gymnasium.envs.safety_gym_v2.builder import *  # noqa
