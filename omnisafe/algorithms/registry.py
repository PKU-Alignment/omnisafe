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
"""Registry for algorithms."""

from __future__ import annotations

import inspect
from typing import Any


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name: str) -> None:
        """Initialize an instance of :class:`Registry`."""
        self._name: str = name
        self._module_dict: dict[str, type] = {}

    @property
    def name(self) -> str:
        """Return the name of the registry."""
        return self._name

    def get(self, key: str) -> Any:
        """Get the class that has been registered under the given key."""
        res = self._module_dict.get(key)
        if res is None:
            raise KeyError(f'{key} is not in the {self.name} registry')
        return res

    def _register_module(self, module_class: type) -> None:
        """Register a module.

        Args:
            module_class (type): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        self._module_dict[module_name] = module_class

    def register(self, cls: type) -> type:
        """Register a module class."""
        self._register_module(cls)
        return cls


REGISTRY = Registry('OmniSafe')


register = REGISTRY.register
get = REGISTRY.get
