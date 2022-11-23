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

from copy import deepcopy

from safety_gymnasium.envs.registration import register
from safety_gymnasium.envs.safety_gym_v2.utils import update_dict_from


VERSION = 'v0'

ROBOT_NAMES = ('Point', 'Car')
ROBOT_XMLS = {name: f'xmls/{name.lower()}.xml' for name in ROBOT_NAMES}
BASE_SENSORS = ['accelerometer', 'velocimeter', 'gyro', 'magnetometer']
EXTRA_SENSORS = {
    'Doggo': [
        'touch_ankle_1a',
        'touch_ankle_2a',
        'touch_ankle_3a',
        'touch_ankle_4a',
        'touch_ankle_1b',
        'touch_ankle_2b',
        'touch_ankle_3b',
        'touch_ankle_4b',
    ],
}
ROBOT_OVERRIDES = {
    'Car': {
        'box_size': 0.125,  # Box half-radius size
        'box_keepout': 0.125,  # Box keepout radius for placement
        'box_density': 0.0005,
    },
}

MAKE_VISION_ENVIRONMENTS = True

# ========================================#
# Helper Class for Easy Gym Registration #
# ========================================#


class SafexpEnvBase:
    """Base used to allow for convenient hierarchies of environments"""

    def __init__(self, name='', config={}, prefix='Safety'):
        """init function"""
        self.name = name
        self.config = config
        self.robot_configs = {}
        self.prefix = prefix
        for robot_name in ROBOT_NAMES:
            robot_config = {}
            robot_config['robot_base'] = ROBOT_XMLS[robot_name]
            robot_config['sensors_obs'] = BASE_SENSORS
            if robot_name in EXTRA_SENSORS:
                robot_config['sensors_obs'] = BASE_SENSORS + EXTRA_SENSORS[robot_name]
            if robot_name in ROBOT_OVERRIDES:
                robot_config.update(ROBOT_OVERRIDES[robot_name])
            self.robot_configs[robot_name] = robot_config

    def copy(self, name='', config={}):
        new_config = deepcopy(self.config)
        update_dict_from(new_config, config)
        return SafexpEnvBase(self.name + name, new_config)

    def register(self, name='', config={}):
        # Note: see safety_gym/envs/mujoco.py for an explanation why we're using
        # 'safety_gym.envs.mujoco:Builder' as the entrypoint, instead of
        # 'safety_gym.envs.engine:Builder'.

        for robot_name, robot_config in self.robot_configs.items():
            # Default
            env_name = f'{self.prefix}{robot_name}{self.name + name}-{VERSION}'
            reg_config = deepcopy(self.config)
            if name:
                reg_config['task']['task_id'] = reg_config['task']['task_id'][:-1] + name
            update_dict_from(reg_config, {'task': robot_config})
            update_dict_from(reg_config, config)
            register(
                id=env_name,
                entry_point='safety_gymnasium.envs.safety_gym_v2.mujoco:Builder',
                kwargs={'config': reg_config},
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision: note, these environments are experimental! Correct behavior not guaranteed
                vision_env_name = f'{self.prefix}{robot_name}{self.name + name}Vision-{VERSION}'
                vision_config = {
                    'world': {},
                    'task': {
                        'observe_vision': True,
                        'observation_flatten': False,
                        'vision_render': True,
                    },
                }
                reg_config = deepcopy(reg_config)
                update_dict_from(reg_config, vision_config)
                register(
                    id=vision_env_name,
                    entry_point='safety_gymnasium.envs.safety_gym_v2.mujoco:Builder',
                    kwargs={'config': reg_config},
                )


# #=============================================================================#
# #                                                                             #
# #       Button Environments                                                   #
# #                                                                             #
# #=============================================================================#

# Shared among all (levels 0, 1, 2)
button = {
    'world': {},
    'task': {
        'task_id': 'ButtonLevelX',
    },
}

# bench_button_base = bench_base.copy('Button', button_all)
bench_button_base = SafexpEnvBase('Button', button)
bench_button_base.register('0', {})
bench_button_base.register('1', {})
bench_button_base.register('2', {})


# =============================================================================#
#                                                                             #
#       Push Environments                                                     #
#                                                                             #
# =============================================================================#

# Shared among all (levels 0, 1, 2)
push = {
    'world': {},
    'task': {
        'task_id': 'PushLevelX',
    },
}

# bench_push_base = bench_base.copy('Push', push)
bench_push_base = SafexpEnvBase('Push', push)
bench_push_base.register('0', {})
bench_push_base.register('1', {})
bench_push_base.register('2', {})

# =============================================================================#
#                                                                             #
#       Goal Environments                                                     #
#                                                                             #
# =============================================================================#

# Shared among all (levels 0, 1, 2)

goal = {
    'world': {},
    'task': {
        'task_id': 'GoalLevelX',
    },
}

# bench_goal_base = bench_base.copy('Goal', goal)
bench_goal_base = SafexpEnvBase('Goal', goal)
bench_goal_base.register('0', {})
bench_goal_base.register('1', {})
bench_goal_base.register('2', {})
