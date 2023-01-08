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
"""Safety Gymnasium Environments."""

from copy import deepcopy

from safety_gymnasium.utils.registration import make, register


__all__ = ['register', 'make']

VERSION = 'v0'

ROBOT_NAMES = ('Point', 'Car', 'Racecar')
ROBOT_XMLS = {name: f'assets/xmls/{name.lower()}.xml' for name in ROBOT_NAMES}
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
# Helper Class for Easy Gym Registration  #
# ========================================#

"""Base used to allow for convenient hierarchies of environments"""
PREFIX = 'Safety'
robot_configs = {}

for name in ROBOT_NAMES:
    config = {}
    config['robot_base'] = ROBOT_XMLS[name]
    config['sensors_obs'] = BASE_SENSORS
    if name in EXTRA_SENSORS:
        config['sensors_obs'] = BASE_SENSORS + EXTRA_SENSORS[name]
    if name in ROBOT_OVERRIDES:
        config.update(ROBOT_OVERRIDES[name])
    robot_configs[name] = config


def combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        for robot_name, robot_config in agents.items():
            # Default
            env_name = f'{PREFIX}{robot_name}{task_name}-{VERSION}'
            combined_config = deepcopy(task_config)
            combined_config.update(robot_config)

            register(
                id=env_name,
                entry_point='safety_gymnasium.builder:Builder',
                kwargs={'config': combined_config, 'task_id': env_name},
                max_episode_steps=max_episode_steps,
            )

            if MAKE_VISION_ENVIRONMENTS:
                # Vision: note, these environments are experimental! Correct behavior not guaranteed
                vision_env_name = f'{PREFIX}{robot_name}{task_name}Vision-{VERSION}'
                vision_config = {
                    'observe_vision': True,
                    'observation_flatten': False,
                    'vision_render': True,
                }
                combined_config = deepcopy(combined_config)
                combined_config.update(vision_config)
                register(
                    id=vision_env_name,
                    entry_point='safety_gymnasium.builder:Builder',
                    kwargs={'config': combined_config, 'task_id': env_name},
                    max_episode_steps=max_episode_steps,
                )


# #=============================================================================#
# #                                                                             #
# #       Button Environments                                                   #
# #                                                                             #
# #=============================================================================#

button_tasks = {'Button0': {}, 'Button1': {}, 'Button2': {}}
combine(button_tasks, robot_configs, max_episode_steps=1000)


# =============================================================================#
#                                                                              #
#       Push Environments                                                      #
#                                                                              #
# =============================================================================#

push_tasks = {'Push0': {}, 'Push1': {}, 'Push2': {}}
combine(push_tasks, robot_configs, max_episode_steps=1000)


# =============================================================================#
#                                                                              #
#       Goal Environments                                                      #
#                                                                              #
# =============================================================================#

goal_tasks = {'Goal0': {}, 'Goal1': {}, 'Goal2': {}}
combine(goal_tasks, robot_configs, max_episode_steps=1000)


# =============================================================================#
#                                                                              #
#       Circle Environments                                                    #
#                                                                              #
# =============================================================================#

circle_tasks = {'Circle0': {}, 'Circle1': {}, 'Circle2': {}}
combine(circle_tasks, robot_configs, max_episode_steps=500)


# =============================================================================#
#                                                                              #
#       Run Environments                                                       #
#                                                                              #
# =============================================================================#

run_tasks = {'Run0': {}}
combine(run_tasks, robot_configs, max_episode_steps=500)


# Safety Velocity
# ----------------------------------------
register(
    id='SafetyHalfCheetahVelocity-v4',
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_half_cheetah_velocity:SafetyHalfCheetahVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='SafetyHopperVelocity-v4',
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_hopper_velocity:SafetyHopperVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='SafetySwimmerVelocity-v4',
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_swimmer_velocity:SafetySwimmerVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='SafetyWalker2dVelocity-v4',
    max_episode_steps=1000,
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_walker2d_velocity:SafetyWalker2dVelocityEnv',
)

register(
    id='SafetyAntVelocity-v4',
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_ant_velocity:SafetyAntVelocityEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)
register(
    id='SafetyHumanoidVelocity-v4',
    entry_point='safety_gymnasium.tasks.safety_velocity.safety_humanoid_velocity:SafetyHumanoidVelocityEnv',
    max_episode_steps=1000,
)
