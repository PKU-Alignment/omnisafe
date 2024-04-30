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
"""Interface of control barrier function-based environments."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import Box


@env_register
class BarrierFunctionEnv(CMDP):
    """Interface of control barrier function-based environments.

    .. warning::
        Since environments based on control barrier functions require special judgment and control of environmental dynamics,
        they do not support the use of vectorized environments for parallelization.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = False
    _support_envs: ClassVar[list[str]] = [
        'Pendulum-v1',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: str = 'cpu',
        **kwargs: Any,
    ) -> None:
        """Initialize the environment.

        Args:
            env_id (str): Environment id.
            num_envs (int, optional): Number of environments. Defaults to 1.
            device (torch.device, optional): Device to store the data. Defaults to 'cpu'.

        Keyword Args:
            render_mode (str, optional): The render mode, ranging from ``human``, ``rgb_array``, ``rgb_array_list``.
                Defaults to ``rgb_array``.
            camera_name (str, optional): The camera name.
            camera_id (int, optional): The camera id.
            width (int, optional): The width of the rendered image. Defaults to 256.
            height (int, optional): The height of the rendered image. Defaults to 256.
        """
        super().__init__(env_id)
        self._env_id = env_id
        if num_envs == 1:
            self._env = gymnasium.make(id=env_id, autoreset=False)
            self._env_specific_setting()
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        else:
            raise NotImplementedError('Only support num_envs=1 now.')
        self._device = torch.device(device)
        self._episodic_violation = []
        self._num_envs = num_envs
        self._metadata = self._env.metadata
        self.env_spec_log = {'Metrics/Max_angle_violation': 0.0}

    def _env_specific_setting(self) -> None:
        """Execute some specific setting for environments.

        Some algorithms based on control barrier functions have made fine-tuning adjustments to the environment.
        We have organized these adjustments and encapsulated them in this function.
        """
        if self._env_id == 'Pendulum-v1':
            self._env.unwrapped.max_torque = 15.0
            self._env.unwrapped.max_speed = 60.0
            self._env.unwrapped.action_space = spaces.Box(
                low=-self._env.unwrapped.max_torque,
                high=self._env.unwrapped.max_torque,
                shape=(1,),
            )
            high = np.array([1.0, 1.0, self._env.unwrapped.max_speed])
            self._env.unwrapped.observation_space = spaces.Box(low=-high, high=high)
            self._env.dt = 0.05
            self._env.dynamics_mode = 'Pendulum'

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
    ]:
        """Step the environment.

        .. note::

            OmniSafe use auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode.
            And the true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            cost: Amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, terminated, truncated)
        )
        cost = torch.abs(torch.atan2(obs[1], obs[0])).to(self._device)
        self._episodic_violation.append(cost)

        if 'final_observation' in info:
            info['final_observation'] = np.array(
                [
                    array if array is not None else np.zeros(obs.shape[-1])
                    for array in info['final_observation']
                ],
            )
            info['final_observation'] = torch.as_tensor(
                info['final_observation'],
                dtype=torch.float32,
                device=self._device,
            )

        return obs, reward, cost, terminated, truncated, info

    def spec_log(self, logger: Logger) -> None:
        """Log specific environment into logger.

        Max angle violation in one episode.

        .. note::
            This function will be called after each episode.

        Args:
            logger (Logger): The logger to use for logging.
        """
        logger.store({'Metrics/Max_angle_violation': max(self._episodic_violation)})
        self._episodic_violation = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        obs, info = self._env.reset(seed=seed, options=options)
        if self._env_id == 'Pendulum-v1':
            while self._env.unwrapped.state[0] > 1.0 or self._env.unwrapped.state[0] < -1.0:
                obs, info = self._env.reset(options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Render the environment.

        Returns:
            Rendered environment.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()

    @property
    def unwrapped(self) -> gymnasium.Env:
        return self._env.unwrapped
