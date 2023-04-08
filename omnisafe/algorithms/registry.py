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
"""Registry for algorithms."""

import inspect


class Registry:
    """A registry to map strings to classes.

    Args:
        name (str): Registry name.
    """

    def __init__(self, name) -> None:
        self._name = name
        self._module_dict: dict = {}

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__ }(name={self._name}, items={list(self._module_dict.keys())})'
        )

    @property
    def name(self):
        """Return the name of the registry."""
        return self._name

    @property
    def module_dict(self):
        """Return a dict mapping names to classes."""
        return self._module_dict

    def get(self, key):
        """Get the class that has been registered under the given key."""
        res = self._module_dict.get(key, None)
        if res is None:
            raise KeyError(f'{key} is not in the {self.name} registry')
        return res

    def _register_module(self, module_class):
        """Register a module.
        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError(f'module must be a class, but got {type(module_class)}')
        module_name = module_class.__name__
        if module_name in self._module_dict:
            raise KeyError(f'{module_name} is already registered in {self.name}')
        self._module_dict[module_name] = module_class

    def register(self, cls):
        """Register a module class."""
        self._register_module(cls)
        return cls


REGISTRY = Registry('OmniSafe')


register = REGISTRY.register
get = REGISTRY.get
