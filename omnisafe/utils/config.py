# Copyright 2022-2032 OmniSafe Team. All Rights Reserved.
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
"""Implementation of Config."""

import json
import os
from typing import Any, Dict, List

import yaml

from omnisafe.typing import Activation, AdvatageEstimator, InitFunction


class Config(dict):
    """Config class for storing hyperparameters."""

    seed: int
    device: str
    device_id: int
    wrapper_type: str
    epochs: int
    steps_per_epoch: int
    actor_iters: int
    critic_iters: int
    check_freq: int
    save_freq: int
    entropy_coef: float
    max_ep_len: int
    num_mini_batches: int
    actor_lr: float
    critic_lr: float
    data_dir: str
    target_kl: float
    batch_size: int
    use_cost: bool
    cost_gamma: float
    linear_lr_decay: bool
    exploration_noise_anneal: bool
    penalty_param: float
    kl_early_stop: bool
    use_max_grad_norm: bool
    max_grad_norm: float
    use_critic_norm: bool
    critic_norm_coeff: bool
    model_cfgs: 'Config'
    shared_weights: bool
    weight_initialization_mode: InitFunction
    actor_type: str
    ac_kwargs: 'Config'
    pi: 'Config'
    hidden_sizes: List[int]
    activation: Activation
    output_activation: Activation
    scale_action: bool
    clip_action: bool
    std_learning: bool
    std_init: float
    val: 'Config'
    num_critics: int
    buffer_cfgs: 'Config'
    gamma: float
    lam: float
    lam_c: float
    adv_eastimator: AdvatageEstimator
    standardized_rew_adv: bool
    standardized_cost_adv: bool
    env_cfgs: 'Config'
    num_envs: int
    async_env: bool
    normalized_rew: bool
    normalized_cost: bool
    normalized_obs: bool
    max_len: int
    num_threads: int

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Config."""
        for key, value in kwargs.items():
            if isinstance(value, dict):
                self[key] = Config.dict2config(value)
            else:
                self[key] = value

    def __getattr__(self, name: str) -> Any:
        """Get attribute."""
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)

    def todict(self) -> dict:
        """Convert Config to dictionary."""
        config_dict = {}
        for key, value in self.items():
            if isinstance(value, Config):
                config_dict[key] = value.todict()
            else:
                config_dict[key] = value
        return config_dict

    def tojson(self) -> str:
        """Convert Config to json string."""
        return json.dumps(self.todict())

    @staticmethod
    def dict2config(config_dict: dict) -> 'Config':
        """Convert dictionary to Config."""
        config = Config()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config[key] = Config.dict2config(value)
            else:
                config[key] = value
        return config

    def recurisve_update(self, update_args: Dict[str, Any]) -> None:
        """Recursively update args."""
        for key, value in self.items():
            if key in update_args:
                if isinstance(update_args[key], dict):
                    if isinstance(value, Config):
                        value.recurisve_update(update_args[key])
                        self[key] = value
                    else:
                        self[key] = Config.dict2config(update_args[key])
                else:
                    self[key] = update_args[key]
        for key, value in update_args.items():
            if key not in self:
                if isinstance(value, dict):
                    self[key] = Config.dict2config(value)
                else:
                    self[key] = value


def get_default_kwargs_yaml(algo: str, env_id: str, algo_type: str) -> Config:
    """Get the default kwargs from ``yaml`` file.

    .. note::
        This function search the ``yaml`` file by the algorithm name and environment name.
        Make sure your new implemented algorithm or environment has the same name as the yaml file.

    Args:
        algo (str): algorithm name.
        env_id (str): environment name.
    """
    path = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(path, '..', 'configs', algo_type, f'{algo}.yaml')
    with open(cfg_path, encoding='utf-8') as file:
        try:
            kwargs = yaml.load(file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, f'{algo}.yaml error: {exc}'
    default_kwargs = kwargs['defaults']
    env_spec_kwargs = kwargs[env_id] if env_id in kwargs.keys() else None

    default_kwargs = Config.dict2config(default_kwargs)

    if env_spec_kwargs is not None:
        default_kwargs.recurisve_update(env_spec_kwargs)

    return default_kwargs


def check_all_configs(configs: Config, algo_type: str) -> None:
    """Check all configs.

    This function is used to check the configs.

    .. note::

        For on-policy algorithms.

        - pi_iters and critic_iters must be greater than 0.
        - actor_lr and critic_lr must be greater than 0.
        - gamma must be in [0, 1).
        - if use_cost is False, cost_gamma must be 1.0.

        For off-policy algorithms.

        - actor_lr and critic_lr must be greater than 0.
        - replay_buffer size must be greater than batch_size.
        - update_every must be less than steps_per_epoch.

    Args:
        configs (dict): configs to be checked.
        algo_type (str): algorithm type.
    """
    __check_env_configs(configs)
    if algo_type == 'on-policy':
        __check_buffer_configs(configs.buffer_cfgs)
        assert configs.actor_iters > 0, 'actor_iters must be greater than 0'
        assert (
            configs.actor_lr > 0 and configs.critic_lr > 0
        ), 'actor_lr and critic_lr must be greater than 0'
        assert (
            configs.buffer_cfgs.gamma >= 0 and configs.buffer_cfgs.gamma < 1.0
        ), 'gamma must be in [0, 1)'
        assert (
            configs.use_cost is False and configs.cost_gamma == 1.0
        ) or configs.use_cost, 'if use_cost is False, cost_gamma must be 1.0'
    elif algo_type == 'off-policy':
        assert (
            configs.actor_lr > 0 and configs.critic_lr > 0
        ), 'actor_lr and critic_lr must be greater than 0'
        assert (
            configs.replay_buffer_cfgs.size > configs.replay_buffer_cfgs.batch_size
        ), 'replay_buffer size must be greater than batch_size'
        assert (
            configs.update_every < configs.steps_per_epoch
        ), 'update_every must be less than steps_per_epoch'


def __check_env_configs(configs: Config) -> None:
    """Check env configs."""
    wrapper_type = configs.wrapper_type
    env_configs = configs.env_cfgs
    assert env_configs.max_len > 0, 'max_len must be greater than 0'
    if wrapper_type == 'SafetyLayerWrapper':
        assert hasattr(
            env_configs, 'safety_layer_cfgs'
        ), 'SafetyLayerWrapper must have safety_layer_cfgs'
    elif wrapper_type == 'SauteWrapper':
        assert (
            hasattr(env_configs, 'unsafe_reward')
            and hasattr(env_configs, 'safety_budget')
            and hasattr(env_configs, 'saute_gamma')
            and hasattr(env_configs, 'scale_safety_budget')
        ), 'SauteWrapper must have unsafe_reward, safety_budget, saute_gamma, scale_safety_budget'
        assert env_configs.unsafe_reward <= 0, 'unsafe_reward must be less or equal than 0'
        assert env_configs.safety_budget > 0, 'safety_budget must be greater than 0'
        assert (
            env_configs.saute_gamma >= 0 and env_configs.saute_gamma < 1.0
        ), 'saute_gamma must be in [0, 1)'
    elif wrapper_type == 'SimmerWrapper':
        assert (
            hasattr(env_configs, 'unsafe_reward')
            and hasattr(env_configs, 'lower_budget')
            and hasattr(env_configs, 'simmer_gamma')
            and hasattr(env_configs, 'scale_safety_budget')
        ), 'SimmerWrapper must have unsafe_reward, safety_budget, simmer_gamma, scale_safety_budget'
        assert env_configs.unsafe_reward <= 0, 'unsafe_reward must be less or equal than 0'
        assert env_configs.lower_budget > 0, 'safety_budget must be greater than 0'
        assert (
            env_configs.simmer_gamma >= 0 and env_configs.simmer_gamma < 1.0
        ), 'simmer_gamma must be in [0, 1)'


def __check_buffer_configs(configs: Config) -> None:
    """Check buffer configs."""
    assert (
        configs.gamma >= 0 and configs.gamma < 1.0
    ), f'gamma must be in [0, 1) but got {configs.gamma}'
    assert configs.lam >= 0 and configs.lam < 1.0, f'lam must be in [0, 1) but got {configs.lam}'
    assert (
        configs.lam_c >= 0 and configs.lam_c < 1.0
    ), f'gamma must be in [0, 1) but got {configs.lam_c}'
    assert configs.adv_estimation_method in ['gae', 'gae-rtg', 'vtrace', 'plain']
