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
"""Implementation of Config."""

import json
import os
from typing import Any, Dict, List

import yaml

from omnisafe.typing import Activation, ActorType, AdvatageEstimator, InitFunction


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
    log_dir: str
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
    model_cfgs: 'ModelConfig'
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

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute."""
        self[name] = value

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
        return json.dumps(self.todict(), indent=4)

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


class ModelConfig(Config):
    """Model config."""

    weight_initialization_mode: InitFunction
    actor_type: ActorType
    actor: 'ModelConfig'
    critic: 'ModelConfig'
    hidden_sizes: List[int]
    activation: Activation
    std: List[float]
    use_obs_encoder: bool
    lr: float


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

    ## check algo configs
    __check_algo_configs(configs.algo_cfgs, algo_type)
    __check_logger_configs(configs.logger_cfgs, algo_type)


def __check_algo_configs(configs: Config, algo_type) -> None:
    """Check algorithm configs."""
    if algo_type == 'onpolicy':
        assert (
            isinstance(configs.update_iters, int) and configs.update_iters > 0
        ), 'update_iters must be int and greater than 0'
        assert (
            isinstance(configs.update_cycle, int) and configs.update_cycle > 0
        ), 'update_cycle must be int and greater than 0'
        assert (
            isinstance(configs.batch_size, int) and configs.batch_size > 0
        ), 'batch_size must be int and greater than 0'
        assert (
            isinstance(configs.target_kl, float) and configs.target_kl >= 0.0
        ), 'target_kl must be float and greater than 0.0'
        assert (
            isinstance(configs.entropy_coef, float)
            and configs.entropy_coef >= 0.0
            and configs.entropy_coef <= 1.0
        ), 'entropy_coef must be float, and it values must be [0.0, 1.0]'
        assert (
            configs.reward_normalize and configs.reward_normalize and configs.reward_normalize
        ), 'normalize must be bool'
        assert isinstance(configs.kl_early_stop, bool), 'kl_early_stop must be bool'
        assert configs.use_max_grad_norm and configs.use_critic_norm, 'norm must be bool'
        assert isinstance(configs.max_grad_norm, float) and isinstance(
            configs.critic_norm_coef, float
        ), 'norm must be bool'
        assert (
            isinstance(configs.gamma, float) and configs.gamma >= 0.0 and configs.gamma <= 1.0
        ), 'gamma must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.cost_gamma, float)
            and configs.cost_gamma >= 0.0
            and configs.cost_gamma <= 1.0
        ), 'cost_gamma must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.lam, float) and configs.lam >= 0.0 and configs.lam <= 1.0
        ), 'lam must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.lam_c, float) and configs.lam_c >= 0.0 and configs.lam_c <= 1.0
        ), 'lam_c must be float, and it values must be [0.0, 1.0]'
        assert (
            isinstance(configs.clip, float) and configs.clip >= 0.0
        ), 'clip must be float, and it values must be [0.0, infty]'
        assert isinstance(configs.adv_estimation_method, str) and configs.adv_estimation_method in [
            'gae',
            'gae-rtg',
            'vtrace',
            'plain',
        ], "adv_estimation_method must be string, and it values must be ['gae','gae-rtg','vtrace','plain']"
        assert (
            configs.standardized_rew_adv and configs.standardized_cost_adv
        ), 'standardized_<>_adv must be bool'
        assert (
            isinstance(configs.penalty_coef, float)
            and configs.penalty_coef >= 0.0
            and configs.penalty_coef <= 1.0
        ), 'penalty_coef must be float, and it values must be [0.0, 1.0]'
        assert isinstance(configs.use_cost, bool), 'penalty_coef must be bool'


def __check_logger_configs(configs: Config, algo_type) -> None:
    """Check logger configs."""
    if algo_type == 'onpolicy':
        assert isinstance(configs.use_wandb, bool) and isinstance(
            configs.wandb_project, str
        ), 'use_wandb and wandb_project must be bool and string'
        assert isinstance(configs.use_tensorboard, bool), 'use_tensorboard must be bool'
        assert isinstance(configs.save_model_freq, int) and isinstance(
            configs.window_lens, int
        ), 'save_model_freq and window_lens must be int'
        assert isinstance(configs.log_dir, str), 'log_dir must be string'
