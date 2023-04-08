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

from __future__ import annotations

import json
import os
from typing import Any

from omnisafe.typing import Activation, ActorType, AdvatageEstimator, InitFunction
from omnisafe.utils.tools import load_yaml


class Config(dict):
    """Config class for storing hyperparameters.

    OmniSafe uses a Config class to store all hyperparameters.
    OmniSafe store hyperparameters in a yaml file and load them into a Config object.
    Then the Config class will check the hyperparameters are valid,
    then pass them to the algorithm class.

    Attributes:
        seed (int): Random seed.
        device (str): Device to use for training.
        device_id (int): Device id to use for training.
        wrapper_type (str): Wrapper type.
        epochs (int): Number of epochs.
        steps_per_epoch (int): Number of steps per epoch.
        actor_iters (int): Number of actor iterations.
        critic_iters (int): Number of critic iterations.
        check_freq (int): Frequency of checking.
        save_freq (int): Frequency of saving.
        entropy_coef (float): Entropy coefficient.
        max_ep_len (int): Maximum episode length.
        num_mini_batches (int): Number of mini batches.
        actor_lr (float): Actor learning rate.
        critic_lr (float): Critic learning rate.
        log_dir (str): Log directory.
        target_kl (float): Target KL divergence.
        batch_size (int): Batch size.
        use_cost (bool): Whether to use cost.
        cost_gamma (float): Cost gamma.
        linear_lr_decay (bool): Whether to use linear learning rate decay.
        exploration_noise_anneal (bool): Whether to use exploration noise anneal.
        penalty_param (float): Penalty parameter.
        kl_early_stop (bool): Whether to use KL early stop.
        use_max_grad_norm (bool): Whether to use max gradient norm.
        max_grad_norm (float): Max gradient norm.
        use_critic_norm (bool): Whether to use critic norm.
        critic_norm_coeff (bool): Critic norm coefficient.
        model_cfgs (ModelConfig): Model config.
        buffer_cfgs (Config): Buffer config.
        gamma (float): Discount factor.
        lam (float): Lambda.
        lam_c (float): Lambda for cost.
        adv_eastimator (AdvatageEstimator): Advantage estimator.
        standardized_rew_adv (bool): Whether to use standardized reward advantage.
        standardized_cost_adv (bool): Whether to use standardized cost advantage.
        env_cfgs (Config): Environment config.
        num_envs (int): Number of environments.
        async_env (bool): Whether to use asynchronous environments.
        env_name (str): Environment name.
        env_kwargs (dict): Environment keyword arguments.
        normalize_obs (bool): Whether to normalize observation.
        normalize_rew (bool): Whether to normalize reward.
        normalize_cost (bool): Whether to normalize cost.
        max_len (int): Maximum length.
        num_threads (int): Number of threads.
    """

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
    model_cfgs: ModelConfig
    buffer_cfgs: Config
    gamma: float
    lam: float
    lam_c: float
    adv_eastimator: AdvatageEstimator
    standardized_rew_adv: bool
    standardized_cost_adv: bool
    env_cfgs: Config
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
    def dict2config(config_dict: dict) -> Config:
        """Convert dictionary to Config.

        Args:
            config_dict (dict): dictionary to be converted.
        """
        config = Config()
        for key, value in config_dict.items():
            if isinstance(value, dict):
                config[key] = Config.dict2config(value)
            else:
                config[key] = value
        return config

    def recurisve_update(self, update_args: dict[str, Any]) -> None:
        """Recursively update args.

        Args:
            update_args (Dict[str, Any]): args to be updated.
        """
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
    actor: ModelConfig
    critic: ModelConfig
    hidden_sizes: list[int]
    activation: Activation
    std: list[float]
    use_obs_encoder: bool
    lr: float | None


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
    print(f'Loading {algo}.yaml from {cfg_path}')
    kwargs = load_yaml(cfg_path)
    default_kwargs = kwargs['defaults']
    env_spec_kwargs = kwargs[env_id] if env_id in kwargs else None

    default_kwargs = Config.dict2config(default_kwargs)

    if env_spec_kwargs is not None:
        default_kwargs.recurisve_update(env_spec_kwargs)

    return default_kwargs


def check_all_configs(configs: Config, algo_type: str) -> None:
    """Check all configs.

    This function is used to check the configs.

    Args:
        configs (dict): configs to be checked.
        algo_type (str): algorithm type.
    """

    ## check algo configs
    __check_algo_configs(configs.algo_cfgs, algo_type)
    __check_logger_configs(configs.logger_cfgs, algo_type)


def __check_algo_configs(configs: Config, algo_type) -> None:
    r"""Check algorithm configs.


    This function is used to check the algorithm configs.

    .. note::

        - ``update_iters`` must be greater than 0 and must be int.
        - ``update_cycle`` must be greater than 0 and must be int.
        - ``batch_size`` must be greater than 0 and must be int.
        - ``target_kl`` must be greater than 0 and must be float.
        - ``entropy_coeff`` must be in [0, 1] and must be float.
        - ``gamma`` must be in [0, 1] and must be float.
        - ``cost_gamma`` must be in [0, 1] and must be float.
        - ``lam`` must be in [0, 1] and must be float.
        - ``lam_c`` must be in [0, 1] and must be float.
        - ``clip`` must be greater than 0 and must be float.
        - ``penalty_coeff`` must be greater than 0 and must be float.
        - ``reward_normalize`` must be bool.
        - ``cost_normalize`` must be bool.
        - ``obs_normalize`` must be bool.
        - ``kl_early_stop`` must be bool.
        - ``use_max_grad_norm`` must be bool.
        - ``use_cost`` must be bool.
        - ``max_grad_norm`` must be greater than 0 and must be float.
        - ``adv_estimation_method`` must be in [``gae``, ``v-trace``, ``gae-rtg``, ``plain``].
        - ``standardized_rew_adv`` must be bool.
        - ``standardized_cost_adv`` must be bool.

    Args:
        configs (Config): configs to be checked.
        algo_type (str): algorithm type.
    """
    if algo_type == 'on-policy':
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
        assert isinstance(configs.reward_normalize, bool), 'reward_normalize must be bool'
        assert isinstance(configs.cost_normalize, bool), 'cost_normalize must be bool'
        assert isinstance(configs.obs_normalize, bool), 'obs_normalize must be bool'
        assert isinstance(configs.kl_early_stop, bool), 'kl_early_stop must be bool'
        assert isinstance(configs.use_max_grad_norm, bool), 'use_max_grad_norm must be bool'
        assert isinstance(configs.use_critic_norm, bool), 'use_critic_norm must be bool'
        assert isinstance(configs.max_grad_norm, float) and isinstance(
            configs.critic_norm_coef,
            float,
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
        if hasattr(configs, 'clip'):
            assert (
                isinstance(configs.clip, float) and configs.clip >= 0.0
            ), 'clip must be float, and it values must be [0.0, infty]'
        assert isinstance(configs.adv_estimation_method, str) and configs.adv_estimation_method in [
            'gae',
            'gae-rtg',
            'vtrace',
            'plain',
        ], "adv_estimation_method must be string, and it values must be ['gae','gae-rtg','vtrace','plain']"
        assert isinstance(configs.standardized_rew_adv, bool) and isinstance(
            configs.standardized_cost_adv,
            bool,
        ), 'standardized_<>_adv must be bool'
        assert (
            isinstance(configs.penalty_coef, float)
            and configs.penalty_coef >= 0.0
            and configs.penalty_coef <= 1.0
        ), 'penalty_coef must be float, and it values must be [0.0, 1.0]'
        assert isinstance(configs.use_cost, bool), 'penalty_coef must be bool'


def __check_logger_configs(configs: Config, algo_type) -> None:
    """Check logger configs."""
    if algo_type == 'on-policy':
        assert isinstance(configs.use_wandb, bool) and isinstance(
            configs.wandb_project,
            str,
        ), 'use_wandb and wandb_project must be bool and string'
        assert isinstance(configs.use_tensorboard, bool), 'use_tensorboard must be bool'
        assert isinstance(configs.save_model_freq, int) and isinstance(
            configs.window_lens,
            int,
        ), 'save_model_freq and window_lens must be int'
        assert isinstance(configs.log_dir, str), 'log_dir must be string'
