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

import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical

import helpers
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.models.actor_critic import ActorCritic
from omnisafe.utils.config_utils import create_namedtuple_from_dict


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    shared=[None],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
)
def test_critic(obs_dim: int, act_dim, shared, hidden_sizes: int, activation: str) -> None:
    ac_kwargs = {'hidden_sizes': [hidden_sizes, hidden_sizes], 'activation': activation}
    builder = CriticBuilder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        shared=shared,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act = torch.randn(act_dim, dtype=torch.float32)
    q_critic = builder.build_critic(critic_type='q')
    v_critic = builder.build_critic(critic_type='v')
    out1 = q_critic(obs, act)
    out2 = v_critic(obs)
    assert out1.shape == torch.Size([]), f'q_critic output shape is {out1.shape}'
    assert out2.shape == torch.Size([]), f'v_critic output shape is {out2.shape}'


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
    weight_initialization_mode=['kaiming_uniform'],
    shared=[None],
)
def test_CategoricalActor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    shared,
) -> None:
    """Test the MLP Categorical Actor class."""
    builder = ActorBuilder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=shared,
    )
    actor = builder.build_actor('categorical')

    obs = torch.randn(obs_dim, dtype=torch.float32)
    dist = actor(obs)
    assert isinstance(dist, Categorical), f'Actor output is not a Categorical distribution'

    act = dist.sample()
    dist, logp = actor(obs, act)
    assert isinstance(dist, Categorical), f'Actor output is not a Categorical distribution'
    assert logp.shape == torch.Size([]), f'Actor logp output shape is {logp.shape}'

    act = actor.predict(obs)
    assert act.shape == torch.Size([]), f'Actor predict output shape is {act.shape}'

    act = actor.predict(obs, deterministic=True)
    assert act.shape == torch.Size([]), f'Actor predict output shape is {act.shape}'

    act, logp = actor.predict(obs, deterministic=True, need_log_prob=True)
    assert act.shape == torch.Size([]), f'Actor predict output shape is {act.shape}'
    assert logp.shape == torch.Size([]), f'Actor logp output shape is {logp.shape}'


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
    weight_initialization_mode=['kaiming_uniform'],
    shared=[None],
    actor_type=['gaussian_annealing', 'gaussian_learning', 'gaussian_stdnet'],
)
def test_GaussianActor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    shared,
    actor_type: str,
) -> None:
    """Test the MLP Gaussian Actor class."""
    builder = ActorBuilder(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=shared,
    )
    kwargs = {
        'act_min': torch.full((act_dim,), -1.0),
        'act_max': torch.full((act_dim,), 1.0),
    }

    actor = builder.build_actor(actor_type=actor_type, **kwargs)

    obs = torch.randn(obs_dim, dtype=torch.float32)
    dist = actor(obs)
    assert isinstance(
        dist, torch.distributions.Normal
    ), f'Actor output is not a Normal distribution'

    act = dist.sample()
    dist, logp = actor(obs, act)
    assert isinstance(
        dist, torch.distributions.Normal
    ), f'Actor output is not a Normal distribution'
    assert logp.shape == torch.Size([]), f'Actor logp output shape is {logp.shape}'

    act = actor.predict(obs)
    assert act.shape == torch.Size([act_dim]), f'Actor predict output shape is {act.shape}'

    act = actor.predict(obs, deterministic=True)
    assert act.shape == torch.Size([act_dim]), f'Actor predict output shape is {act.shape}'
    act, logp = actor.predict(obs, deterministic=True, need_log_prob=True)
    assert act.shape == torch.Size([act_dim]), f'Actor predict output shape is {act.shape}'
    assert logp.shape == torch.Size([]), f'Actor logp output shape is {logp.shape}'


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    space_type=[Box, Discrete],
    standardized_obs=[True, False],
    scale_rewards=[True, False],
    shared_weights=[False],  # shared weights not implemented yet in discrete case.
    hidden_sizes=[64],
    activation=['relu'],
    weight_initialization_mode=['kaiming_uniform'],
    actor_type=['gaussian_annealing', 'gaussian_learning', 'gaussian_stdnet'],
)
def test_ActorCritic(
    obs_dim: int,
    act_dim: int,
    space_type,
    standardized_obs: bool,
    scale_rewards: bool,
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
            'actor_type': actor_type,
        },
        'val': {
            'hidden_sizes': [hidden_sizes, hidden_sizes],
            'activation': activation,
        },
    }
    observation_space = Box(low=-1, high=1, shape=(obs_dim,))

    model_cfgs = {
        'ac_kwargs': ac_kwargs,
        'weight_initialization_mode': weight_initialization_mode,
        'shared_weights': shared_weights,
    }
    model_cfgs = create_namedtuple_from_dict(model_cfgs)

    if space_type == Discrete:
        action_space = space_type(act_dim)
    else:
        action_space = space_type(low=-1, high=1, shape=(act_dim,))

    actor_critic = ActorCritic(
        observation_space=observation_space,
        action_space=action_space,
        standardized_obs=standardized_obs,
        scale_rewards=scale_rewards,
        model_cfgs=model_cfgs,
    )

    obs = torch.randn(obs_dim, dtype=torch.float32)

    act, val, logpro = actor_critic(obs)
    assert (
        isinstance(act, np.ndarray)
        and isinstance(val, np.ndarray)
        and isinstance(logpro, np.ndarray)
    ), 'Failed!'

    act, val, logpro = actor_critic.step(obs)
    assert (
        isinstance(act, np.ndarray)
        and isinstance(val, np.ndarray)
        and isinstance(logpro, np.ndarray)
    ), 'Failed!'

    act, val, logpro = actor_critic.step(obs, deterministic=True)
    assert (
        isinstance(act, np.ndarray)
        and isinstance(val, np.ndarray)
        and isinstance(logpro, np.ndarray)
    ), 'Failed!'

    # TODO: Test anneal_exploration method.
