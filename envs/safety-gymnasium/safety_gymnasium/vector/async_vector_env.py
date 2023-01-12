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
"""An async vector environment."""

import multiprocessing as mp
import sys
from copy import deepcopy
from typing import List, Optional, Sequence, Tuple, Union

import gymnasium
import numpy as np
from gymnasium.error import NoAsyncCallError
from gymnasium.vector.async_vector_env import AsyncState, AsyncVectorEnv
from gymnasium.vector.utils import concatenate, write_to_shared_memory
from safety_gymnasium.vector.utils.tile_images import tile_images


__all__ = ['AsyncVectorEnv']


class SafetyAsyncVectorEnv(AsyncVectorEnv):
    """The async vectorized environment for safety gymnasium."""

    # pylint: disable-next = too-many-arguments
    def __init__(
        self,
        env_fns: Sequence[callable],
        observation_space: Optional[gymnasium.Space] = None,
        action_space: Optional[gymnasium.Space] = None,
        shared_memory: bool = True,
        copy: bool = True,
        context: Optional[str] = None,
        daemon: bool = True,
        worker: Optional[callable] = None,
    ) -> None:
        """The initialization."""
        target = _worker_shared_memory if shared_memory else _worker
        target = worker or target
        super().__init__(
            env_fns,
            observation_space,
            action_space,
            shared_memory,
            copy,
            context,
            daemon,
            worker=target,
        )

    def get_images(self):
        """get the images from the environment."""
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.parent_pipes]
        return imgs

    def render(self):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        return bigimg

    # pylint: disable-next = too-many-locals
    def step_wait(
        self, timeout: Optional[Union[int, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """Wait for the calls to :obj:`step` in each sub-environment to finish.

        Args:
            timeout: Number of seconds before the call to :meth:`step_wait`
            times out. If ``None``, the call to :meth:`step_wait` never times out.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                'Calling `step_wait` without any prior call to `step_async`.',
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f'The call to `step_wait` has timed out after {timeout} second(s).'
            )

        observations_list, rewards, costs, terminateds, truncateds, infos = [], [], [], [], [], {}
        successes = []
        for i, pipe in enumerate(self.parent_pipes):
            result, success = pipe.recv()
            obs, rew, cost, terminated, truncated, info = result

            successes.append(success)
            observations_list.append(obs)
            rewards.append(rew)
            costs.append(cost)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos = self._add_info(infos, info, i)

        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                self.single_observation_space,
                observations_list,
                self.observations,
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(costs),
            np.array(terminateds, dtype=np.bool_),
            np.array(truncateds, dtype=np.bool_),
            infos,
        )


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches
def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation, info = env.reset(**data)
                pipe.send(((observation, info), True))

            elif command == 'step':
                (
                    observation,
                    reward,
                    cost,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info['final_observation'] = old_observation
                    info['final_info'] = old_info
                pipe.send(((observation, reward, cost, terminated, truncated, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'render':
                pipe.send((env.render()))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'seed', 'close']:
                    raise ValueError(
                        f'Trying to call function `{name}` with '
                        f'`_call`. Use `{name}` directly instead.'
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == '_setattr':
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                pipe.send(
                    (
                        (data[0] == env.observation_space, data[1] == env.action_space),
                        True,
                    )
                )
            else:
                raise RuntimeError(
                    f'Received unknown command `{command}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, `render`, `_call`, '
                    '`_setattr`, `_check_spaces`}.'
                )
    # pylint: disable-next = broad-except
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation, info = env.reset(**data)
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, info), True))
            elif command == 'step':
                (
                    observation,
                    reward,
                    cost,
                    terminated,
                    truncated,
                    info,
                ) = env.step(data)
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info['final_observation'] = old_observation
                    info['final_info'] = old_info
                write_to_shared_memory(observation_space, index, observation, shared_memory)
                pipe.send(((None, reward, cost, terminated, truncated, info), True))
            elif command == 'render':
                pipe.send((env.render()))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_call':
                name, args, kwargs = data
                if name in ['reset', 'step', 'seed', 'close']:
                    raise ValueError(
                        f'Trying to call function `{name}` with '
                        f'`_call`. Use `{name}` directly instead.'
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == '_setattr':
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == '_check_spaces':
                pipe.send(((data[0] == observation_space, data[1] == env.action_space), True))
            else:
                raise RuntimeError(
                    f'Received unknown command `{command}`. Must '
                    'be one of {`reset`, `step`, `seed`, `close`, `render`, `_call`, '
                    '`_setattr`, `_check_spaces`}.'
                )
    # pylint: disable-next = broad-except
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
