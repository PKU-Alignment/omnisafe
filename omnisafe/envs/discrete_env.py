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
"""Environments with discrete observation/action space in Gymnasium."""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium
import numpy as np
import torch
from gymnasium import spaces

from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU, Discrete


@env_register
class DiscreteEnv(CMDP):
    """Discrete Gymnasium Environment.

    This environment only served as an example to integrate discrete action and
    observation environment into OmniSafe. We support
    `CartPole-v1 <https://gymnasium.farama.org/environments/classic_control/cart_pole/>`_
    and `Taxi-v3 <https://gymnasium.farama.org/environments/toy_text/taxi/>`_.
    The former is ``Box`` observation space and ``Discrete`` action space, while
    the latter is ``Discrete`` observation and ``Discrete`` action space.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
        need_action_repeat_wrapper (bool): Whether to use action repeat wrapper.
        need_action_scale_wrapper (bool): Whether to use action scale wrapper.
    """

    need_action_scale_wrapper = False
    need_obs_normalize_wrapper = False
    need_auto_reset_wrapper = False
    need_time_limit_wrapper = False

    _support_envs: ClassVar[list[str]] = [
        'CartPole-v1',
        'Taxi-v3',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,
    ) -> None:
        """Initialize an instance of :class:`DiscreteEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if num_envs > 1:
            self._env = gymnasium.vector.make(
                id=env_id,
                num_envs=num_envs,
                render_mode=kwargs.get('render_mode'),
            )
            assert isinstance(
                self._env.single_action_space,
                Discrete,
            ), 'Only support Discrete action space.'
            self._action_space = self._env.single_action_space
            self._observation_space = self._env.single_observation_space  # type: ignore
        else:
            self.need_time_limit_wrapper = True
            self.need_auto_reset_wrapper = True
            self._env = gymnasium.make(id=env_id, autoreset=True, render_mode=kwargs.get('render_mode'))  # type: ignore
            self._action_space = self._env.action_space  # type: ignore
            self._observation_space = self._env.observation_space  # type: ignore
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
        obs, reward, terminated, truncated, info = self._env.step(
            action.detach().cpu().squeeze().numpy(),
        )
        obs, reward, terminated, truncated = (
            torch.as_tensor(x, dtype=torch.float32, device=self._device)
            for x in (obs, reward, terminated, truncated)
        )
        if isinstance(self._observation_space, spaces.Discrete):
            obs = obs.unsqueeze(-1)
        if 'final_observation' in info:
            if isinstance(info['final_observation'], np.ndarray):
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
            if isinstance(self._observation_space, spaces.Discrete):
                info['final_observation'] = info['final_observation'].unsqueeze(-1)

        return obs, reward, torch.zeros_like(reward), terminated, truncated, info

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
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self._device)
        if isinstance(self._observation_space, spaces.Discrete):
            obs = obs.unsqueeze(-1)
        return obs, info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset(seed=seed)

    def sample_action(self) -> torch.Tensor:
        """Sample a random action.

        Returns:
            A random action.
        """
        return torch.as_tensor(
            self._env.action_space.sample(),
            dtype=torch.int64,
            device=self._device,
        )

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
