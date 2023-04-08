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
"""Typing utilities."""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
from gymnasium.spaces import Box, Discrete
from torch.types import Device


RenderFrame = TypeVar('RenderFrame')
OmnisafeSpace = Union[Box, Discrete]
Activation = Literal['identity', 'relu', 'sigmoid', 'softplus', 'tanh']
AdvatageEstimator = Literal['gae', 'gae-rtg', 'vtrace', 'plain']
InitFunction = Literal['kaiming_uniform', 'xavier_normal', 'glorot', 'xavier_uniform', 'orthogonal']
CriticType = Literal['v', 'q']
ActorType = Literal['gaussian_learning', 'gaussian_sac', 'mlp']
cpu = torch.device('cpu')


__all__ = [
    'Activation',
    'AdvatageEstimator',
    'InitFunction',
    'Callable',
    'List',
    'Optional',
    'Sequence',
    'Tuple',
    'TypeVar',
    'Union',
    'Dict',
    'NamedTuple',
    'Any',
    'OmnisafeSpace',
    'RenderFrame',
    'Device',
    'cpu',
]
