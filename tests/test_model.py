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
"""Test models."""

import pytest
import torch
from gymnasium.spaces import Box, Discrete

import helpers
from omnisafe.models import ActorBuilder, CriticBuilder
from omnisafe.models.actor_critic import ActorCritic, ConstraintActorCritic
from omnisafe.typing import Activation
from omnisafe.utils.config import Config


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    hidden_sizes=[64],
    activation=['tanh', 'relu'],
    use_obs_encoder=[True, False],
    num_critics=[1, 2],
)
def test_critic(
    obs_dim: int,
    act_dim: int,
    num_critics: int,
    hidden_sizes: int,
    activation: Activation,
    use_obs_encoder: bool,
) -> None:
    """Test critic."""
    obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = Box(low=-1.0, high=1.0, shape=(act_dim,))

    builder = CriticBuilder(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        num_critics=num_critics,
        use_obs_encoder=use_obs_encoder,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act = torch.randn(act_dim, dtype=torch.float32)
    q_critic = builder.build_critic(critic_type='q')
    v_critic = builder.build_critic(critic_type='v')
    with pytest.raises(NotImplementedError):
        builder.build_critic(critic_type='invalid')

    out1 = q_critic(obs, act)[0]
    out2 = v_critic(obs)[0]
    assert out1.shape == torch.Size([]), f'q_critic output shape is {out1.shape}'
    assert out2.shape == torch.Size([]), f'v_critic output shape is {out2.shape}'


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    hidden_sizes=[64],
    activation=['tanh', 'relu'],
    deterministic=[True, False],
)
def test_actor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: Activation,
    deterministic: bool,
) -> None:
    """Test actor."""
    obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = Box(low=-1.0, high=1.0, shape=(act_dim,))

    builder = ActorBuilder(
        obs_space=obs_space,
        act_space=act_space,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    actor_learning = builder.build_actor(actor_type='gaussian_learning')
    actor_sac = builder.build_actor(actor_type='gaussian_sac')
    actor_mlp = builder.build_actor(actor_type='mlp')
    with pytest.raises(NotImplementedError):
        builder.build_actor(actor_type='invalid')

    _ = actor_learning(obs)
    action = actor_learning.predict(obs, deterministic)
    assert action.shape == torch.Size([act_dim]), f'actor output shape is {action.shape}'
    logp = actor_learning.log_prob(action)
    assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    actor_learning.std = 0.9
    assert (actor_learning.std - 0.9) < 1e-4, f'actor std is {actor_learning.std}'

    _ = actor_sac(obs)
    action = actor_sac.predict(obs, deterministic)
    assert action.shape == torch.Size([act_dim]), f'actor output shape is {action.shape}'
    logp = actor_sac.log_prob(action)
    assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    with pytest.raises(NotImplementedError):
        actor_sac.std = 0.9
    assert isinstance(actor_sac.std, float), f'actor std is {actor_sac.std}'

    action = actor_mlp.predict(obs, deterministic)
    actor_mlp.noise = 0.1
    assert actor_mlp.noise == 0.1, f'actor noise is {actor_mlp.noise}'
    with pytest.raises(AssertionError):
        actor_mlp.noise = -0.1
    with pytest.raises(NotImplementedError):
        actor_mlp(obs)
    with pytest.raises(NotImplementedError):
        actor_mlp.log_prob(action)


@helpers.parametrize(
    linear_lr_decay=[True, False],
    lr=[None, 1e-3],
)
def test_actor_critic(
    linear_lr_decay: bool,
    lr,
):
    """Test actor critic."""
    obs_dim = 10
    act_dim = 5
    obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,))
    act_space = Box(low=-1.0, high=1.0, shape=(act_dim,))

    model_cfgs = Config(
        weight_initialization_mode='kaiming_uniform',
        actor_type='gaussian_learning',
        linear_lr_decay=linear_lr_decay,
        exploration_noise_anneal=False,
        std_range=[0.5, 0.1],
        actor=Config(hidden_sizes=[64, 64], activation='tanh', lr=lr),
        critic=Config(hidden_sizes=[64, 64], activation='tanh', lr=lr),
    )

    ac = ActorCritic(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=model_cfgs,
        epochs=10,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act, value, logp = ac(obs)
    assert act.shape == torch.Size([act_dim]), f'actor output shape is {act.shape}'
    assert value.shape == torch.Size([]), f'critic output shape is {value.shape}'
    assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    ac.set_annealing(epochs=[1, 10], std=[0.5, 0.1])
    ac.annealing(5)

    cac = ConstraintActorCritic(
        obs_space=obs_space,
        act_space=act_space,
        model_cfgs=model_cfgs,
        epochs=10,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    act, value_r, value_c, logp = cac(obs)
    assert act.shape == torch.Size([act_dim]), f'actor output shape is {act.shape}'
    assert value_r.shape == torch.Size([]), f'critic output shape is {value_r.shape}'
    assert value_c.shape == torch.Size([]), f'critic output shape is {value_c.shape}'
    assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
    cac.set_annealing(epochs=[1, 10], std=[0.5, 0.1])
    cac.annealing(5)


@helpers.parametrize(
    obs_dim=[10],
    act_dim=[5],
    hidden_sizes=[64],
    activation=['tanh', 'relu'],
    deterministic=[True, False],
)
def test_discrete_actor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: Activation,
    deterministic: bool,
) -> None:
    """Test actor."""
    box_obs_space = Box(low=-1.0, high=1.0, shape=(obs_dim,))
    # discrete_obs_space = Discrete(1)
    act_space = Discrete(act_dim)

    builder = ActorBuilder(
        obs_space=box_obs_space,
        act_space=act_space,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    actor_discrete = builder.build_actor(actor_type='discrete')
    with pytest.raises(NotImplementedError):
        builder.build_actor(actor_type='invalid')

    _ = actor_discrete(obs)
    action = actor_discrete.predict(obs, deterministic)
    assert action.shape == torch.Size([1, 1]), f'actor output shape is {action.shape}'
    logp = actor_discrete.log_prob(action)
    assert logp.shape == torch.Size([]), f'actor log_prob shape is {logp.shape}'
