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
"""The sync vectored environment."""

from copy import deepcopy
from typing import Callable, Iterator

import numpy as np
from gymnasium import Env
from gymnasium.spaces import Space
from gymnasium.vector.sync_vector_env import SyncVectorEnv
from gymnasium.vector.utils import concatenate
from safety_gymnasium.vector.utils.tile_images import tile_images


__all__ = ['SafetySyncVectorEnv']


class SafetySyncVectorEnv(SyncVectorEnv):
    """Vectored safe environment that serially runs multiple safe environments."""

    def __init__(
        self,
        env_fns: Iterator[Callable[[], Env]],
        observation_space: Space = None,
        action_space: Space = None,
        copy: bool = True,
    ):
        super().__init__(env_fns, observation_space, action_space, copy)
        self._costs = np.zeros((self.num_envs,), dtype=np.float64)

    def render(self):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        return bigimg

    def step_wait(self):
        """Steps through each of the environments returning the batched results.

        Returns:
            The batched environment step results
        """
        observations, infos = [], {}
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):

            (
                observation,
                self._rewards[i],
                self._costs[i],
                self._terminateds[i],
                self._truncateds[i],
                info,
            ) = env.step(action)

            if self._terminateds[i] or self._truncateds[i]:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info['final_observation'] = old_observation
                info['final_info'] = old_info
            observations.append(observation)
            infos = self._add_info(infos, info, i)
        self.observations = concatenate(
            self.single_observation_space, observations, self.observations
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._costs),
            np.copy(self._terminateds),
            np.copy(self._truncateds),
            infos,
        )

    def get_images(self):
        """Get images from child environments."""
        return [env.render('rgb_array') for env in self.envs]
