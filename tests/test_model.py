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
"""Test models"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete

import helpers
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.models.actor_critic import ActorCritic
from omnisafe.models.actor_q_critic import ActorQCritic
from omnisafe.utils.config_utils import dict2namedtuple
from omnisafe.utils.model_utils import Activation, InitFunction


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    shared=[None],
    hidden_sizes=[64],
    activation=['tanh', 'relu'],
    use_obs_encoder=[True, False],
)
def test_critic(
    obs_dim: int,
    act_dim,
    shared,
    hidden_sizes: int,
    activation: str,
    use_obs_encoder: bool,
) -> None:
    """Test critic"""
    builder = CriticBuilder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        shared=shared,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act = torch.randn(act_dim, dtype=torch.float32)
    q_critic = builder.build_critic(critic_type='q', use_obs_encoder=use_obs_encoder)
    v_critic = builder.build_critic(critic_type='v')
    out1 = q_critic(obs, act)[0]
    out2 = v_critic(obs)
    assert out1.shape == torch.Size([]), f'q_critic output shape is {out1.shape}'
    assert out2.shape == torch.Size([]), f'v_critic output shape is {out2.shape}'


@helpers.parametrize(
    actor_type=['gaussian', 'gaussian_stdnet'],
    obs_dim=[10],
    act_dim=[5],
    hidden_sizes=[64],
    activation=['tanh'],
    output_activation=['tanh'],
    weight_initialization_mode=['kaiming_uniform'],
    shared=[None],
    std_learning=[True],
    std_init=[1.0],
    scale_action=[True],
    clip_action=[True],
)
def test_gaussian_actor(
    actor_type: str,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: list,
    activation: Activation,
    weight_initialization_mode: InitFunction,
    shared: nn.Module,
    scale_action: bool,
    clip_action: bool,
    output_activation: Optional[Activation],
    std_learning: bool,
    std_init: float,
) -> None:
    """Test the MLP Gaussian Actor class."""
    builder = ActorBuilder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=shared,
        scale_action=scale_action,
        clip_action=clip_action,
        output_activation=output_activation,
        std_learning=std_learning,
        std_init=std_init,
    )
    kwargs = {
        'act_min': torch.full((act_dim,), -1.0),
        'act_max': torch.full((act_dim,), 1.0),
    }

    actor = builder.build_actor(actor_type=actor_type, **kwargs)

    obs = torch.randn((1, obs_dim), dtype=torch.float32)
    dist = actor(obs)
    assert isinstance(dist, torch.distributions.Normal), 'Actor output is not a Normal distribution'

    raw_act, act = actor.predict(obs)
    assert act.shape == torch.Size([1, act_dim]), f'Actor predict output shape is {act.shape}'
    assert raw_act.shape == torch.Size(
        [1, act_dim]
    ), f'Actor predict output shape is {raw_act.shape}'

    raw_act, act = actor.predict(obs, deterministic=True)
    assert act.shape == torch.Size([1, act_dim]), f'Actor predict output shape is {act.shape}'
    assert raw_act.shape == torch.Size(
        [1, act_dim]
    ), f'Actor predict output shape is {raw_act.shape}'
    raw_act, act, logp = actor.predict(obs, deterministic=True, need_log_prob=True)

    assert raw_act.shape == torch.Size(
        [1, act_dim]
    ), f'Actor predict output shape is {raw_act.shape}'
    assert act.shape == torch.Size([1, act_dim]), f'Actor predict output shape is {act.shape}'
    assert logp.shape == torch.Size([1]), f'Actor logp output shape is {logp.shape}'


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    space_type=[Box, Discrete],
    shared_weights=[False, True],  # shared weights not implemented yet in discrete case.
    hidden_sizes=[64],
    activation=['tanh'],
    weight_initialization_mode=[
        'kaiming_uniform',
        'xavier_normal',
        'glorot',
        'xavier_uniform',
        'orthogonal',
    ],
    actor_type=['gaussian', 'gaussian_stdnet'],
)
def test_actor_critic(
    obs_dim: int,
    act_dim: int,
    space_type,
    shared_weights: bool,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    actor_type: str,
) -> None:
    """Test the Actor Critic class."""

    ac_kwargs = {
        'pi': {
            'hidden_sizes': [hidden_sizes, hidden_sizes],
            'activation': activation,
        },
        'val': {
            'hidden_sizes': [hidden_sizes, hidden_sizes],
            'activation': activation,
        },
    }
    observation_space = Box(low=-1, high=1, shape=(obs_dim,))

    model_cfgs = dict2namedtuple(
        {
            'actor_type': actor_type,
            'ac_kwargs': ac_kwargs,
            'weight_initialization_mode': weight_initialization_mode,
            'shared_weights': shared_weights,
        }
    )

    if space_type == Discrete:
        action_space = space_type(act_dim)
    else:
        action_space = space_type(low=-1, high=1, shape=(act_dim,))

    actor_critic = ActorCritic(
        observation_space=observation_space,
        action_space=action_space,
        model_cfgs=model_cfgs,
    )

    obs = torch.randn((1, obs_dim), dtype=torch.float32)

    raw_act, act, val, logpro = actor_critic(obs)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    raw_act, act, val, logpro = actor_critic.step(obs)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    raw_act, act, val, logpro = actor_critic.step(obs, deterministic=True)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    actor_critic.anneal_exploration(0.5)


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    space_type=[Box, Discrete],
    shared_weights=[False],  # shared weights not implemented yet in discrete case.
    hidden_sizes=[64],
    activation=['tanh'],
    weight_initialization_mode=[
        'kaiming_uniform',
        'xavier_normal',
        'glorot',
        'xavier_uniform',
        'orthogonal',
    ],
    actor_type=['gaussian', 'gaussian_stdnet'],
)
def test_actor_q_critic(
    obs_dim: int,
    act_dim: int,
    space_type,
    shared_weights: bool,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    actor_type: str,
) -> None:
    """Test the Actor Critic class."""

    ac_kwargs = {
        'pi': {
            'hidden_sizes': [hidden_sizes, hidden_sizes],
            'activation': activation,
        },
        'val': {
            'hidden_sizes': [hidden_sizes, hidden_sizes],
            'activation': activation,
            'num_critics': 1,
        },
    }
    observation_space = Box(low=-1, high=1, shape=(obs_dim,))

    model_cfgs = dict2namedtuple(
        {
            'actor_type': actor_type,
            'ac_kwargs': ac_kwargs,
            'weight_initialization_mode': weight_initialization_mode,
            'shared_weights': shared_weights,
        }
    )

    if space_type == Discrete:
        action_space = space_type(act_dim)
    else:
        action_space = space_type(low=-1, high=1, shape=(act_dim,))

    actor_critic = ActorQCritic(
        observation_space=observation_space,
        action_space=action_space,
        model_cfgs=model_cfgs,
    )

    obs = torch.randn((1, obs_dim), dtype=torch.float32)

    raw_act, act, val, logpro = actor_critic(obs)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    raw_act, act, val, logpro = actor_critic.step(obs)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    raw_act, act, val, logpro = actor_critic.step(obs, deterministic=True)
    assert (
        isinstance(raw_act, torch.Tensor)
        and isinstance(act, torch.Tensor)
        and isinstance(val, torch.Tensor)
        and isinstance(logpro, torch.Tensor)
    ), 'Failed!'

    actor_critic.anneal_exploration(0.5)
