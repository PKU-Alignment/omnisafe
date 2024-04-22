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
"""Test registry."""

import pytest

from omnisafe.algorithms.registry import Registry
from omnisafe.envs.core import CMDP, env_register, env_unregister


class TestRegistry:
    name: str = 'test'
    idx: int = 0


def test_with_error() -> None:
    registry = Registry('test')
    TestRegistry()
    with pytest.raises(TypeError):
        registry.register('test')
    with pytest.raises(KeyError):
        registry.get('test')
    with pytest.raises(TypeError):

        @env_register
        class TestEnv:
            name: str = 'test'
            idx: int = 0

    with pytest.raises(ValueError):

        @env_register
        class CustomEnv(CMDP):
            pass

    @env_register
    @env_unregister
    @env_unregister
    class CustomEnv(CMDP):  # noqa
        _support_envs = ['Simple-v0']  # noqa
