# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Environment API for OmniSafe."""

from omnisafe.envs import classic_control
from omnisafe.envs.core import CMDP, env_register, make, support_envs
from omnisafe.envs.crabs_env import CRABSEnv
from omnisafe.envs.custom_env import CustomEnv
from omnisafe.envs.meta_drive_env import SafetyMetaDriveEnv
from omnisafe.envs.mujoco_env import MujocoEnv
from omnisafe.envs.safety_gymnasium_env import SafetyGymnasiumEnv
from omnisafe.envs.safety_gymnasium_modelbased import SafetyGymnasiumModelBased
from omnisafe.envs.safety_isaac_gym_env import SafetyIsaacGymEnv
