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


from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Any, List, Type
import inspect

import torch

from omnisafe.typing import OmnisafeSpace


class CMDP(ABC):
    _support_envs: List[str]
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace

    @classmethod
    def support_envs(cls) -> List[str]:
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
        assert env_id in self.support_envs(), f"env_id {env_id} is not supported by {self.__class__.__name__}"

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

    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Run one timestep of the environment's dynamics using the agent actions.

        Args:
            action (torch.Tensor): action.

        Returns:
            observation (torch.Tensor): agent's observation of the current environment.
            reward (torch.Tensor): amount of reward returned after previous action.
            cost (torch.Tensor): amount of cost returned after previous action.
            terminated (torch.Tensor): whether the episode has ended, in which case further step() calls will return undefined results.
            truncated (torch.Tensor): whether the episode has been truncated due to a time limit.
            info (Dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[torch.Tensor, Dict]:
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

    @abstractmethod
    def close(self) -> None:
        """Close the environment.
        """


class EnvRegister:
    def __init__(self) -> None:
        self._class: Dict[str, Type[CMDP]] = {}
        self._support_envs: Dict[str, List[str]] = {}

    def _register(self, env_class: Type[CMDP]) -> None:
        """Register the environment class.

        Args:
            env_class (Type[CMDP]): the environment class.
        """
        if not inspect.isclass(env_class):
            raise TypeError(f"{env_class} must be a class")
        class_name = env_class.__name__
        if not issubclass(env_class, CMDP):
            raise TypeError(f"{class_name} must be subclass of CMDP")
        if class_name in self._class:
            raise ValueError(f"{class_name} has been registered")
        env_ids = env_class.support_envs()
        self._class[class_name] = env_class
        self._support_envs[class_name] = env_ids

    def register(self, env_class: Type[CMDP]) -> Type[CMDP]:
        """Register the environment class.

        Args:
            env_class (Type[CMDP]): the environment class.

        Returns:
            Type[CMDP]: the environment class.
        """
        self._register(env_class)
        return env_class

    def get_class(self, env_id: str, class_name: Optional[str]) -> Type[CMDP]:
        """Get the environment class.

        Args:
            env_id (str): the environment id.
            class_name (Optional[str]): the environment class name.

        Returns:
            Type[CMDP]: the environment class.
        """
        if class_name is not None:
            assert class_name in self._class, f"{class_name} is not registered"
            assert env_id in self._support_envs[class_name], f"{env_id} is not supported by {class_name}"
            return self._class[class_name]
        else:
            for class_name, env_ids in self._support_envs.items():
                if env_id in env_ids:
                    return self._class[class_name]
            raise ValueError(f"{env_id} is not supported by any environment class")

    def support_envs(self) -> List[str]:
        """The supported environments.

        Returns:
            List[str]: the supported environments.
        """
        return list(set([env_id for env_ids in self._support_envs.values() for env_id in env_ids]))


ENV_REGISTRY = EnvRegister()

env_register = ENV_REGISTRY.register
support_envs = ENV_REGISTRY.support_envs


def make(env_id: str, class_name: Optional[str] = None, **kwargs) -> CMDP:
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
    "CMDP",
    "env_register",
    "support_envs",
    "make",
]
