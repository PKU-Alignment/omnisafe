# Copyright 2022 OmniSafe Team. All Rights Reserved.
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
"""Register and make environments."""

import copy
import importlib
import importlib.util
from typing import Optional, Sequence, Union

from gymnasium import Env, error, logger
from gymnasium.envs.registration import (  # pylint: disable=unused-import
    EnvSpec,
    _check_version_exists,
    find_highest_version,
    get_env_id,
    load,
    namespace,
    parse_env_id,
)
from gymnasium.envs.registration import register as gymnasium_register
from gymnasium.envs.registration import registry, spec  # pylint: disable=unused-import
from gymnasium.wrappers import AutoResetWrapper, HumanRendering, OrderEnforcing, RenderCollection
from gymnasium.wrappers.compatibility import EnvCompatibility
from safety_gymnasium.wrappers import SafePassiveEnvChecker, SafeTimeLimit


safe_registry = set()


def register(**kwargs):
    """Register an environment."""
    safe_registry.add(kwargs['id'])
    gymnasium_register(**kwargs)


# pylint: disable-next=too-many-arguments,too-many-branches,too-many-statements,too-many-locals
def make(
    id: Union[str, EnvSpec],  # pylint: disable=redefined-builtin
    max_episode_steps: Optional[int] = None,
    autoreset: bool = False,
    apply_api_compatibility: Optional[bool] = None,
    disable_env_checker: Optional[bool] = None,
    **kwargs,
) -> Env:
    """Create an environment according to the given ID.

    To find all available environments use `gymnasium.envs.registry.keys()` for all valid ids.

    Args:
        id: Name of the environment. Optionally, a module to import can be included, eg. 'module:Env-v0'
        max_episode_steps: Maximum length of an episode (TimeLimit wrapper).
        autoreset: Whether to automatically reset the environment after each episode (AutoResetWrapper).
        apply_api_compatibility: Whether to wrap the environment with the `StepAPICompatibility` wrapper that
            converts the environment step from a done bool to return termination and truncation bools.
            By default, the argument is None to which the environment specification `apply_api_compatibility` is used
            which defaults to False. Otherwise, the value of `apply_api_compatibility` is used.
            If `True`, the wrapper is applied otherwise, the wrapper is not applied.
        disable_env_checker: If to run the env checker,
            None will default to the environment specification `disable_env_checker`
            (which is by default False, running the environment checker),
            otherwise will run according to this parameter (`True` = not run, `False` = run)
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        Error: If the ``id`` doesn't exist then an error is raised
    """
    if isinstance(id, EnvSpec):
        spec_ = id
    else:
        module, id = (None, id) if ':' not in id else id.split(':')
        if module is not None:
            try:
                importlib.import_module(module)
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    f'{e}. Environment registration via importing a module failed. '
                    f"Check whether '{module}' contains env registration and can be imported."
                ) from e
        spec_ = registry.get(id)

        name_space, name, version = parse_env_id(id)
        latest_version = find_highest_version(name_space, name)
        if version is not None and latest_version is not None and latest_version > version:
            logger.warn(
                f'The environment {id} is out of date. You should consider '
                f'upgrading to version `v{latest_version}`.'
            )
        if version is None and latest_version is not None:
            version = latest_version
            new_env_id = get_env_id(name_space, name, version)
            spec_ = registry.get(new_env_id)
            logger.warn(
                f'Using the latest versioned environment `{new_env_id}` '
                f'instead of the unversioned environment `{id}`.'
            )

        if spec_ is None:
            _check_version_exists(name_space, name, version)
            raise error.Error(f'No registered env with id: {id}')

        assert id in safe_registry, f'Environment {id} is not registered in safety-gymnasium.'

    _kwargs = spec_.kwargs.copy()
    _kwargs.update(kwargs)

    if spec_.entry_point is None:
        raise error.Error(f'{spec_.id} registered but entry_point is not specified')
    if callable(spec_.entry_point):
        env_creator = spec_.entry_point
    else:
        # Assume it's a string
        env_creator = load(spec_.entry_point)

    mode = _kwargs.get('render_mode')
    apply_human_rendering = False
    apply_render_collection = False

    # If we have access to metadata we check that "render_mode" is valid and
    # see if the HumanRendering wrapper needs to be applied
    if mode is not None and hasattr(env_creator, 'metadata'):
        assert isinstance(
            env_creator.metadata, dict
        ), f'Expect the environment creator ({env_creator}) metadata to be dict,\
                actual type: {type(env_creator.metadata)}'

        if 'render_modes' in env_creator.metadata:
            render_modes = env_creator.metadata['render_modes']
            if not isinstance(render_modes, Sequence):
                logger.warn(
                    f'Expects the environment metadata render_modes to be a Sequence (tuple or list),\
                        actual type: {type(render_modes)}'
                )

            # Apply the `HumanRendering` wrapper, if the mode=="human" but "human" not in render_modes
            if (
                mode == 'human'
                and 'human' not in render_modes
                and ('rgb_array' in render_modes or 'rgb_array_list' in render_modes)
            ):
                logger.warn(
                    "You are trying to use 'human' rendering for an environment that doesn't natively support it. "
                    'The HumanRendering wrapper is being applied to your environment.'
                )
                apply_human_rendering = True
                if 'rgb_array' in render_modes:
                    _kwargs['render_mode'] = 'rgb_array'
                else:
                    _kwargs['render_mode'] = 'rgb_array_list'
            elif (
                mode not in render_modes
                and mode.endswith('_list')
                and mode[: -len('_list')] in render_modes
            ):
                _kwargs['render_mode'] = mode[: -len('_list')]
                apply_render_collection = True
            elif mode not in render_modes:
                logger.warn(
                    f'The environment is being initialised with mode ({mode})\
                        that is not in the possible render_modes ({render_modes}).'
                )
        else:
            logger.warn(
                f"The environment creator metadata doesn't include `render_modes`,\
                    contains: {list(env_creator.metadata.keys())}"
            )

    if apply_api_compatibility is True or (
        apply_api_compatibility is None and spec_.apply_api_compatibility is True
    ):
        # If we use the compatibility layer, we treat the render mode explicitly and don't pass it to the env creator
        render_mode = _kwargs.pop('render_mode', None)
    else:
        render_mode = None

    try:
        env = env_creator(**_kwargs)
    except TypeError as e:
        if (
            str(e).find("got an unexpected keyword argument 'render_mode'") >= 0
            and apply_human_rendering
        ):
            raise error.Error(
                f"You passed render_mode='human' although {id} doesn't implement human-rendering natively. "
                'Gym tried to apply the HumanRendering wrapper but it looks like your environment is using the old '
                'rendering API, which is not supported by the HumanRendering wrapper.'
            )

        raise e

    # Copies the environment creation specification and kwargs to add to the environment specification details
    spec_ = copy.deepcopy(spec_)
    spec_.kwargs = _kwargs
    env.unwrapped.spec = spec_

    # Add step API wrapper
    if apply_api_compatibility is True or (
        apply_api_compatibility is None and spec_.apply_api_compatibility is True
    ):
        env = EnvCompatibility(env, render_mode)

    # Run the environment checker as the lowest level wrapper
    if disable_env_checker is False or (
        disable_env_checker is None and spec_.disable_env_checker is False
    ):
        env = SafePassiveEnvChecker(env)

    # Add the order enforcing wrapper
    if spec_.order_enforce:
        env = OrderEnforcing(env)

    # Add the time limit wrapper
    if max_episode_steps is not None:
        env = SafeTimeLimit(env, max_episode_steps)
    elif spec_.max_episode_steps is not None:
        env = SafeTimeLimit(env, spec_.max_episode_steps)

    # Add the autoreset wrapper
    if autoreset:
        env = AutoResetWrapper(env)

    # Add human rendering wrapper
    if apply_human_rendering:
        env = HumanRendering(env)
    elif apply_render_collection:
        env = RenderCollection(env)

    return env
