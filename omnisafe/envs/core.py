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
"""The core module of the environment."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, ClassVar

import torch

from omnisafe.typing import DEVICE_CPU, OmnisafeSpace


__all__ = [
    'CMDP',
    'Wrapper',
    'env_register',
    'support_envs',
    'make',
]


class CMDP(ABC):
    """The core class of the environment.

    The CMDP class is the core class of the environment. It defines the basic interface of the
    environment. The environment should inherit from this class and implement the abstract methods.

    Attributes:
        need_time_limit_wrapper (bool): Whether the environment need time limit wrapper.
        need_auto_reset_wrapper (bool): Whether the environment need auto reset wrapper.
    """

    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    _metadata: dict[str, Any]

    _num_envs: int
    _time_limit: int | None = None
    need_time_limit_wrapper: bool
    need_auto_reset_wrapper: bool
    need_action_scale_wrapper: bool

    _support_envs: ClassVar[list[str]]

    @classmethod
    def support_envs(cls) -> list[str]:
        """The supported environments.

        Returns:
            The supported environments.
        """
        return cls._support_envs

    @abstractmethod
    def __init__(self, env_id: str, **kwargs: Any) -> None:
        """Initialize an instance of :class:`CMDP`."""
        assert (
            env_id in self.support_envs()
        ), f'env_id {env_id} is not supported by {self.__class__.__name__}'

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment."""
        return self._action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment."""
        return self._observation_space

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of the environment."""
        return self._metadata

    @property
    def num_envs(self) -> int:
        """The number of parallel environments."""
        return self._num_envs

    @property
    def time_limit(self) -> int | None:
        """The time limit of the environment."""
        return self._time_limit

    @abstractmethod
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
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """

    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.

        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set the seed for this env's random number generator(s).

        Args:
            seed (int): The seed to use.
        """

    @abstractmethod
    def sample_action(self) -> torch.Tensor:
        """Sample an action from the action space.

        Returns:
            The sampled action.
        """

    @abstractmethod
    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components.
        """
        return {}

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""


class Wrapper(CMDP):
    """The wrapper class of the environment.

    The Wrapper class is the wrapper class of the environment. It defines the basic interface of the
    environment wrapper. The environment wrapper should inherit from this class and implement the
    abstract methods.

    Args:
        env (CMDP): The environment.
        device (torch.device): The device to use. Defaults to ``torch.device('cpu')``.

    Attributes:
        _env (CMDP): The environment.
    """

    def __init__(self, env: CMDP, device: torch.device = DEVICE_CPU) -> None:
        """Initialize an instance of :class:`Wrapper`."""
        self._env: CMDP = env
        self._device: torch.device = device

    def __getattr__(self, name: str) -> Any:
        """Get the attribute of the environment.

        Args:
            name (str): The attribute name.

        Returns:
            The attribute.
        """
        if name.startswith('_'):
            raise AttributeError(f'attempted to get missing private attribute {name}')
        return getattr(self._env, name)

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
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): The action from the agent or random.

        Returns:
            observation: The agent's observation of the current environment.
            reward: The amount of reward returned after previous action.
            cost: The amount of cost returned after previous action.
            terminated: Whether the episode has ended.
            truncated: Whether the episode has been truncated due to a time limit.
            info: Some information logged by the environment.
        """
        return self._env.step(action)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment and returns an initial observation.

        Args:
            seed (int, optional): The random seed. Defaults to None.
            options (dict[str, Any], optional): The options for the environment. Defaults to None.


        Returns:
            observation: The initial observation of the space.
            info: Some information logged by the environment.
        """
        return self._env.reset(seed=seed, options=options)

    def set_seed(self, seed: int) -> None:
        """Set the seed for this env's random number generator(s).

        Args:
            seed (int): The random seed to use.
        """
        self._env.set_seed(seed)

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the action space.

        Returns:
            The sampled action.
        """
        return self._env.sample_action()

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            The render frames: we recommend to use `np.ndarray`
                which could construct video by moviepy.
        """
        return self._env.render()

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        .. note::
            The saved components will be stored in the wrapped environment. If the environment is
            not wrapped, the saved components will be empty dict. common wrappers are obs_normalize,
            reward_normalize, and cost_normalize.

        Returns:
            The saved components.
        """
        return self._env.save()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class EnvRegister:
    """The environment register.

    The EnvRegister is used to register the environment class. It provides the method to get the
    environment class by the environment id.

    Examples:
        >>> from omnisafe.envs.core import env_register
        >>> from cunstom_env import CustomEnv
        >>> @env_register
        ... class CustomEnv:
        ...     ...
    """

    def __init__(self) -> None:
        """Initialize an instance of :class:`EnvRegister`."""
        self._class: dict[str, type[CMDP]]
        self._support_envs: dict[str, list[str]]
        self._class = {}
        self._support_envs = {}

    def _register(self, env_class: type[CMDP]) -> None:
        """Register the environment class.

        Args:
            env_class (type[CMDP]): The environment class.
        """
        if not inspect.isclass(env_class):
            raise TypeError(f'{env_class} must be a class')
        class_name = env_class.__name__
        if not issubclass(env_class, CMDP):
            raise TypeError(f'{class_name} must be subclass of CMDP')
        if class_name in self._class:
            raise ValueError(f'{class_name} has been registered')
        env_ids = env_class.support_envs()
        self._class[class_name] = env_class
        self._support_envs[class_name] = env_ids

    def register(self, env_class: type[CMDP]) -> type[CMDP]:
        """Register the environment class.

        Args:
            env_class (type[CMDP]): The environment class.

        Returns:
            The environment class.
        """
        self._register(env_class)
        return env_class

    def get_class(self, env_id: str, class_name: str | None) -> type[CMDP]:
        """Get the environment class.

        Args:
            env_id (str): The environment id.
            class_name (str or None): The environment class name.

        Returns:
            The environment class.
        """
        if class_name is not None:
            assert class_name in self._class, f'{class_name} is not registered'
            assert (
                env_id in self._support_envs[class_name]
            ), f'{env_id} is not supported by {class_name}'
            return self._class[class_name]

        for cls_name, env_ids in self._support_envs.items():
            if env_id in env_ids:
                return self._class[cls_name]
        raise ValueError(f'{env_id} is not supported by any environment class')

    def support_envs(self) -> list[str]:
        """The supported environments.

        Returns:
            The supported environments.
        """
        return list({env_id for env_ids in self._support_envs.values() for env_id in env_ids})


ENV_REGISTRY = EnvRegister()

env_register = ENV_REGISTRY.register
support_envs = ENV_REGISTRY.support_envs


def make(env_id: str, class_name: str | None = None, **kwargs: Any) -> CMDP:
    """Create an environment.

    Args:
        env_id (str): The environment id.
        class_name (str or None): The environment class name.

    Keyword Args:
        render_mode (str, optional): The render mode ranges from 'human' to 'rgb_array' and 'rgb_array_list'.
            Defaults to 'rgb_array'.
        camera_name (str, optional): The camera name.
        camera_id (int, optional): The camera id.
        width (int, optional): The width of the rendered image. Defaults to 256.
        height (int, optional): The height of the rendered image. Defaults to 256.

    Returns:
        The environment class.
    """
    env_class = ENV_REGISTRY.get_class(env_id, class_name)
    return env_class(env_id, **kwargs)
