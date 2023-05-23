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
"""Implementation of the Base algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from omnisafe.common.logger import Logger
from omnisafe.utils import distributed
from omnisafe.utils.config import Config
from omnisafe.utils.tools import get_device, seed_all


class BaseAlgo(ABC):  # pylint: disable=too-few-public-methods
    """Base class for all algorithms."""

    _logger: Logger

    def __init__(self, env_id: str, cfgs: Config) -> None:
        """Initialize an instance of algorithm."""
        self._env_id: str = env_id
        self._cfgs: Config = cfgs

        assert hasattr(cfgs, 'seed'), 'Please specify the seed in the config file.'
        self._seed: int = int(cfgs.seed) + distributed.get_rank() * 1000
        seed_all(self._seed)

        assert hasattr(cfgs.train_cfgs, 'device'), 'Please specify the device in the config file.'
        self._device: torch.device = get_device(self._cfgs.train_cfgs.device)

        distributed.setup_distributed()

        self._init_env()
        self._init_model()

        self._init()

        self._init_log()

    @property
    def logger(self) -> Logger:
        """Get the logger."""
        return self._logger  # pylint: disable=no-member

    @property
    def cost_limit(self) -> float | None:
        """Get the cost limit."""
        return getattr(self._cfgs.algo_cfgs, '_cost_limit', None)

    @abstractmethod
    def _init(self) -> None:
        """Initialize the algorithm."""

    @abstractmethod
    def _init_env(self) -> None:
        """Initialize the environment."""

    @abstractmethod
    def _init_model(self) -> None:
        """Initialize the model."""

    @abstractmethod
    def _init_log(self) -> None:
        """Initialize the logger."""

    @abstractmethod
    def learn(self) -> tuple[float, float, float]:
        """Learn the policy."""
