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
"""Environments interface of SafeMetaDrive."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np
import torch

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import DEVICE_CPU


META_DRIVE_AVAILABLE = True
try:
    from metadrive import SafeMetaDriveEnv
except ImportError:
    META_DRIVE_AVAILABLE = False


@env_register
class SafetyMetaDriveEnv(CMDP):
    """SafeMetaDrive Environment.

    More information about MetaDrive environment is provided in https://github.com/metadriverse/metadrive.
    For the details of environment configuration, please refer to https://github.com/decisionforce/EGPO.

    Args:
        env_id (str): Environment id.
        num_envs (int, optional): Number of environments. Defaults to 1.
        device (torch.device, optional): Device to store the data. Defaults to
            ``torch.device('cpu')``.

    Keyword Args:
        meta_drive_config (dict, optional): MetaDrive configuration, containing following keys:
            - ``horizon``: Max iterations of interactions.
            - ``random_traffic``: Whether to use random traffic.
            - ``crash_vehicle_penalty``: The penalty when crash into other vehicles.
            - ``crash_object_penalty``: The penalty when crash into other objects.
            - ``out_of_road_penalty``: The penalty when out of road.

    Attributes:
        need_auto_reset_wrapper (bool): Whether to use auto reset wrapper.
        need_time_limit_wrapper (bool): Whether to use time limit wrapper.
    """

    need_auto_reset_wrapper: bool = True
    need_time_limit_wrapper: bool = False
    env_spec_log: dict[str, Any]

    _support_envs: ClassVar[list[str]] = [
        'SafeMetaDrive',
    ]

    def __init__(
        self,
        env_id: str,
        num_envs: int = 1,
        device: torch.device = DEVICE_CPU,
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> None:
        """Initialize an instance of :class:`SafetyMetaDriveEnv`."""
        super().__init__(env_id)
        self._num_envs = num_envs
        self._device = torch.device(device)

        if META_DRIVE_AVAILABLE:
            self._env = SafeMetaDriveEnv(config=kwargs.get('meta_drive_config'))
        else:
            raise ImportError(
                'Please install MetaDrive to use SafeMetaDrive!\
                \nInstall from source: https://github.com/metadriverse/metadrive.\
                \nInstall from PyPI: `pip install metadrive`.',
            )
        self._num_scenarios = self._env.config['num_scenarios']

        self._env.logger.setLevel(logging.FATAL)
        self._action_space = self._env.action_space
        self._observation_space = self._env.observation_space
        self._metadata = self._env.metadata
        self.env_spec_log = {'Env/Success_rate': []}

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
            action.detach().cpu().numpy(),
        )
        cost = info['cost']
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

        # for meta drive environment, log the success rate when terminated
        if terminated or truncated:
            self.env_spec_log['Env/Success_rate'].append(int(info['arrive_dest']))

        return obs, reward, cost, terminated, truncated, info

    def spec_log(self, logger: Logger) -> None:
        """Log the success rate into the logger."""
        logger.store({'Env/Success_rate': np.mean(self.env_spec_log['Env/Success_rate'])})
        self.env_spec_log['Env/Success_rate'] = []

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment.

        Args:
            seed (int or None, optional): Seed to reset the environment.
                Defaults to None.

        Returns:
            observation: Agent's observation of the current environment.
            info: Some information logged by the environment.
        """
        obs, info = self._env.reset(seed=seed)
        return torch.as_tensor(obs, dtype=torch.float32, device=self._device), info

    def set_seed(self, seed: int) -> None:
        """Set the seed for the environment.

        Args:
            seed (int): Seed to set.
        """
        self.reset()

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
