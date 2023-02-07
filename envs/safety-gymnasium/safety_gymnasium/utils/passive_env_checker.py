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
"""A set of functions for passively checking environment implementations."""

import numpy as np
from gymnasium import error, logger
from gymnasium.utils.passive_env_checker import check_obs


def env_step_passive_checker(env, action):
    """A passive check for the environment step, investigating the returning data then returning the data unchanged."""
    # We don't check the action as for some environments then out-of-bounds values can be given
    result = env.step(action)
    assert isinstance(
        result, tuple
    ), f'Expects step result to be a tuple, actual type: {type(result)}'
    if len(result) == 5:
        logger.deprecation(
            'Core environment is written in old step API which returns one bool instead of two. '
            'It is recommended to rewrite the environment with new step API. '
        )
        obs, reward, cost, done, info = result

        if not isinstance(done, (bool, np.bool_)):
            logger.warn(f'Expects `done` signal to be a boolean, actual type: {type(done)}')
    elif len(result) == 6:
        obs, reward, cost, terminated, truncated, info = result

        # np.bool is actual python bool not np boolean type, therefore bool_ or bool8
        if not isinstance(terminated, (bool, np.bool_)):
            logger.warn(
                f'Expects `terminated` signal to be a boolean, actual type: {type(terminated)}'
            )
        if not isinstance(truncated, (bool, np.bool_)):
            logger.warn(
                f'Expects `truncated` signal to be a boolean, actual type: {type(truncated)}'
            )
    else:
        raise error.Error(
            f'Expected `Env.step` to return a four or five element tuple,\
            actual number of elements returned: {len(result)}.'
        )

    check_obs(obs, env.observation_space, 'step')
    check_reward_cost(reward=reward, cost=cost)

    assert isinstance(
        info, dict
    ), f'The `info` returned by `step()` must be a python dictionary, actual type: {type(info)}'

    return result


def check_reward_cost(reward, cost):
    """Check out the type and the value of the reward and cost."""
    if not (np.issubdtype(type(reward), np.integer) or np.issubdtype(type(reward), np.floating)):
        logger.warn(
            f'The reward returned by `step()` must be a float,\
            int, np.integer or np.floating, actual type: {type(reward)}'
        )
    else:
        if np.isnan(reward):
            logger.warn('The reward is a NaN value.')
        if np.isinf(reward):
            logger.warn('The reward is an inf value.')

    if not (np.issubdtype(type(cost), np.integer) or np.issubdtype(type(cost), np.floating)):
        # logger.warn(
        #     'The reward returned by `step()` must be a float,\
        #     int, np.integer or np.floating, actual type: {type(cost)}'
        # )
        pass
    else:
        if np.isnan(cost):
            logger.warn('The reward is a NaN value.')
        if np.isinf(cost):
            logger.warn('The reward is an inf value.')
