# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
from typing import Any

import torch

from omnisafe.typing import OmnisafeSpace, cpu


class CMDP(ABC):
    """The core class of the environment.

    The CMDP class is the core class of the environment. It defines the basic
    interface of the environment. The environment should inherit from this class
    and implement the abstract methods.

    Attributes:
        _support_envs (List[str]): the supported environments.
        _action_space (OmnisafeSpace): the action space of the environment.
        _observation_space (OmnisafeSpace): the observation space of the environment.
        _num_envs (int): the parallel environments, for env that not support parallel, num_envs should be 1
        _time_limit (Optional[int]): the time limit of the environment, if None, the environment is infinite.
        need_time_limit_wrapper (bool): whether the environment need time limit wrapper.
        need_auto_reset_wrapper (bool): whether the environment need auto reset wrapper.
    """

    _support_envs: list[str]
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    _metadata: dict[str, Any]

    _num_envs: int
    _time_limit: int | None = None
    need_time_limit_wrapper: bool
    need_auto_reset_wrapper: bool

    @classmethod
    def support_envs(cls) -> list[str]:
        """The supported environments.

        Returns:
            List[str]: the supported environments.
        """
        return cls._support_envs

    @abstractmethod
    def __init__(self, env_id: str, **kwargs) -> None:
        """Initialize the environment.

        Args:
            env_id (str): the environment id.
        """
        assert (
            env_id in self.support_envs()
        ), f'env_id {env_id} is not supported by {self.__class__.__name__}'

    @property
    def action_space(self) -> OmnisafeSpace:
        """The action space of the environment.

        Returns:
            OmnisafeSpace: the action space.
        """
        return self._action_space

    @property
    def observation_space(self) -> OmnisafeSpace:
        """The observation space of the environment.

        Returns:
            OmnisafeSpace: the observation space.
        """
        return self._observation_space

    @property
    def metadata(self) -> dict[str, Any]:
        """The metadata of the environment.

        Returns:
            Dict[str, Any]: the metadata.
        """
        return self._metadata

    @property
    def num_envs(self) -> int:
        """The parallel environments.

        Returns:
            int: the parallel environments.
        """
        return self._num_envs

    @property
    def time_limit(self) -> int | None:
        """The time limit of the environment.

        Returns:
            Optional[int]: the time limit of the environment.
        """
        return self._time_limit

    @abstractmethod
    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Resets the environment and returns an initial observation.

        Args:
            seed (Optional[int]): seed for the environment.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Sets the seed for this env's random number generator(s).

        Args:
            seed (int): the seed to use.
        """

    @abstractmethod
    def sample_action(self) -> torch.Tensor:
        """Sample an action from the action space.

        Returns:
            torch.Tensor: the sampled action.
        """

    @abstractmethod
    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            Any: the render frames, we recommend to use `np.ndarray` which could construct video by moviepy.
        """

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.
        Returns:
            Dict[str, torch.nn.Module]: the saved components.
        """
        return {}

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""


class Wrapper(CMDP):
    """The wrapper class of the environment.

    The Wrapper class is the wrapper class of the environment. It defines the basic
    interface of the environment wrapper. The environment wrapper should inherit
    from this class and implement the abstract methods.

    Attributes:
        _env (CMDP): the environment.

    """

    def __init__(self, env: CMDP, device: torch.device = cpu) -> None:
        """Initialize the wrapper.

        Args:
            env (CMDP): the environment.

        Attributes:
            _env (CMDP): the environment.
            _device (torch.device): the device of the environment.
        """
        self._env = env
        self._device = device

    def __getattr__(self, name: str) -> Any:
        """Get the attribute of the environment.

        Args:
            name (str): the attribute name.

        Returns:
            Any: the attribute.
        """
        if name.startswith('_'):
            raise AttributeError(f'attempted to get missing private attribute {name}')
        return getattr(self._env, name)

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._env.step(action)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        """Resets the environment and returns an initial observation.

        Args:
            seed (Optional[int]): seed for the environment.

        Returns:
            observation (torch.Tensor): the initial observation of the space.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return self._env.reset(seed)

    def set_seed(self, seed: int) -> None:
        """Sets the seed for this env's random number generator(s).

        Args:
            seed (int): the seed to use.
        """
        self._env.set_seed(seed)

    def sample_action(self) -> torch.Tensor:
        """Sample an action from the action space.

        Returns:
            torch.Tensor: the sampled action.
        """
        return self._env.sample_action()

    def render(self) -> Any:
        """Compute the render frames as specified by :attr:`render_mode` during the initialization of the environment.

        Returns:
            Any: the render frames, we recommend to use `np.ndarray` which could construct video by moviepy.
        """
        return self._env.render()

    def save(self) -> dict[str, torch.nn.Module]:
        """Save the important components of the environment.

        Returns:
            Dict[str, torch.nn.Module]: the saved components.
        """
        return self._env.save()

    def close(self) -> None:
        """Close the environment."""
        self._env.close()


class EnvRegister:
    """The environment register.

    The EnvRegister is used to register the environment class. It provides the
    method to get the environment class by the environment id.

    Example:
        >>> from omnisafe.envs.core import env_register
        >>> from cunstom_env import CustomEnv
        >>> @env_register
        >>> class CustomEnv():

    Attributes:
        _class (Dict[str, Type[CMDP]]): the registered environment class.
        _support_envs (Dict[str, List[str]]): the environment ids supported by the environment class.
    """

    def __init__(self) -> None:
        self._class: dict[str, type[CMDP]] = {}
        self._support_envs: dict[str, list[str]] = {}

    def _register(self, env_class: type[CMDP]) -> None:
        """Register the environment class.

        Args:
            env_class (Type[CMDP]): the environment class.
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
            env_class (Type[CMDP]): the environment class.

        Returns:
            Type[CMDP]: the environment class.
        """
        self._register(env_class)
        return env_class

    def get_class(self, env_id: str, class_name: str | None) -> type[CMDP]:
        """Get the environment class.

        Args:
            env_id (str): the environment id.
            class_name (Optional[str]): the environment class name.

        Returns:
            Type[CMDP]: the environment class.
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
            List[str]: the supported environments.
        """
        return list({env_id for env_ids in self._support_envs.values() for env_id in env_ids})


ENV_REGISTRY = EnvRegister()

env_register = ENV_REGISTRY.register
support_envs = ENV_REGISTRY.support_envs


def make(env_id: str, class_name: str | None = None, **kwargs) -> CMDP:
    """Create an environment.

    Args:
        env_id (str): the environment id.
        class_name (Optional[str]): the environment class name.
        **kwargs: the keyword arguments for the environment initialization.

    Returns:
        CMDP: the environment.
    """
    env_class = ENV_REGISTRY.get_class(env_id, class_name)
    return env_class(env_id, **kwargs)


__all__ = [
    'CMDP',
    'Wrapper',
    'env_register',
    'support_envs',
    'make',
]
