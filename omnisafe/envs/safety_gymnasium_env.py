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
"""Environments in the Safety-Gymnasium."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import safety_gymnasium
import torch

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Box


@env_register
class SafetyGymnasiumEnv(CMDP):
    """Safety Gymnasium Environment.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = False
    need_time_limit_wrapper: bool = False

    _support_envs: ClassVar[list[str]] = [
        'SafetyPointGoal0-v0',
        'SafetyPointGoal1-v0',
        'SafetyPointGoal2-v0',
        'SafetyPointButton0-v0',
        'SafetyPointButton1-v0',
        'SafetyPointButton2-v0',
        'SafetyPointPush0-v0',
        'SafetyPointPush1-v0',
        'SafetyPointPush2-v0',
        'SafetyPointCircle0-v0',
        'SafetyPointCircle1-v0',
        'SafetyPointCircle2-v0',
        'SafetyCarGoal0-v0',
        'SafetyCarGoal1-v0',
        'SafetyCarGoal2-v0',
        'SafetyCarButton0-v0',
        'SafetyCarButton1-v0',
        'SafetyCarButton2-v0',
        'SafetyCarPush0-v0',
        'SafetyCarPush1-v0',
        'SafetyCarPush2-v0',
        'SafetyCarCircle0-v0',
        'SafetyCarCircle1-v0',
        'SafetyCarCircle2-v0',
        'SafetyAntGoal0-v0',
        'SafetyAntGoal1-v0',
        'SafetyAntGoal2-v0',
        'SafetyAntButton0-v0',
        'SafetyAntButton1-v0',
        'SafetyAntButton2-v0',
        'SafetyAntPush0-v0',
        'SafetyAntPush1-v0',
        'SafetyAntPush2-v0',
        'SafetyAntCircle0-v0',
        'SafetyAntCircle1-v0',
        'SafetyAntCircle2-v0',
        'SafetyDoggoGoal0-v0',
        'SafetyDoggoGoal1-v0',
        'SafetyDoggoGoal2-v0',
        'SafetyDoggoButton0-v0',
        'SafetyDoggoButton1-v0',
        'SafetyDoggoButton2-v0',
        'SafetyDoggoPush0-v0',
        'SafetyDoggoPush1-v0',
        'SafetyDoggoPush2-v0',
        'SafetyDoggoCircle0-v0',
        'SafetyDoggoCircle1-v0',
        'SafetyDoggoCircle2-v0',
        'SafetyRacecarGoal0-v0',
        'SafetyRacecarGoal1-v0',
        'SafetyRacecarGoal2-v0',
        'SafetyRacecarButton0-v0',
        'SafetyRacecarButton1-v0',
        'SafetyRacecarButton2-v0',
        'SafetyRacecarPush0-v0',
        'SafetyRacecarPush1-v0',
        'SafetyRacecarPush2-v0',
        'SafetyRacecarCircle0-v0',
        'SafetyRacecarCircle1-v0',
        'SafetyRacecarCircle2-v0',
        'SafetyHalfCheetahVelocity-v1',
        'SafetyHopperVelocity-v1',
        'SafetySwimmerVelocity-v1',
        'SafetyWalker2dVelocity-v1',
        'SafetyAntVelocity-v1',
        'SafetyHumanoidVelocity-v1',
        'SafetyPointRun0-v0',
        'SafetyCarRun0-v0',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`SafetyGymnasiumEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if num_envs > 1:
            self._env = safety_gymnasium.vector.make(env_id=env_id, num_envs=num_envs, **kwargs)
            assert isinstance(self._env.single_action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.single_observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = safety_gymnasium.make(id=env_id, autoreset=False, **kwargs)
            assert isinstance(self._env.action_space, Box), 'Only support Box action space.'
            assert isinstance(
                self._env.observation_space,
                Box,
            ), 'Only support Box observation space.'
            self._action_space = self._env.action_space
            self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata

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
            OmniSafe uses auto reset wrapper to reset the environment when the episode is
            terminated. So the ``obs`` will be the first observation of the next episode. And the
            true ``final_observation`` in ``info`` will be stored in the ``final_observation`` key
            of ``info``.

        Args:
            action (torch.Tensor): Action to take.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        obs, reward, cost, terminated, truncated, info = self._env.step(
            action.detach().cpu().numpy(),
        )
        obs, reward, cost, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, cost, terminated, truncated)
        )
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

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    @property
    def max_episode_steps(self) -> int:
        """The max steps per episode."""
        return self._env.spec.max_episode_steps

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()
