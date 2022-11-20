import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions import Categorical

import helpers
from omnisafe.algos.models.actor_critic import ActorCritic
from omnisafe.algos.models.critic import Critic
from omnisafe.algos.models.mlp_categorical_actor import MLPCategoricalActor
from omnisafe.algos.models.mlp_gaussian_actor import MLPGaussianActor


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    shared=[None],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
)
def test_critic(obs_dim: int, shared, hidden_sizes: int, activation: str) -> None:
    ac_kwargs = {'hidden_sizes': [hidden_sizes, hidden_sizes], 'activation': activation}
    print(ac_kwargs['hidden_sizes'])
    critic = Critic(obs_dim, shared=shared, **ac_kwargs)
    input = torch.rand(obs_dim, dtype=torch.float32)
    out = critic(input)


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
    weight_initialization_mode=[
        'kaiming_uniform',
        'xavier_normal',
        'glorot',
        'xavier_uniform',
        'orthogonal',
    ],
    shared=[None],
)
def test_MLPGaussianActor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    shared,
) -> None:
    """Test the MLP Gaussian Actor class."""
    gaussianActor = MLPGaussianActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=shared,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)
    out, logpro = gaussianActor(obs)
    assert isinstance(out, torch.distributions.normal.Normal) and logpro is None, 'Failed!'
    out, logpro = gaussianActor(obs, torch.tensor(act_dim, dtype=torch.float32))
    assert isinstance(out, torch.distributions.normal.Normal) and isinstance(
        logpro, torch.Tensor
    ), 'Failed!'

    dist = gaussianActor.dist(obs)
    assert isinstance(dist, torch.distributions.normal.Normal), 'Failed'

    act, logpro = gaussianActor.predict(obs)
    assert isinstance(act, torch.Tensor) and isinstance(logpro, torch.Tensor), 'Failed!'


@helpers.parametrize(
    obs_dim=[1, 10, 100],
    act_dim=[1, 5, 10],
    hidden_sizes=[64, 128, 256],
    activation=['tanh', 'softplus', 'sigmoid', 'identity', 'relu'],
    weight_initialization_mode=[
        'kaiming_uniform',
        'xavier_normal',
        'glorot',
        'xavier_uniform',
        'orthogonal',
    ],
    shared=[None],
)
def test_MLPCategoricalActor(
    obs_dim: int,
    act_dim: int,
    hidden_sizes: int,
    activation: str,
    weight_initialization_mode: str,
    shared,
) -> None:
    """Test the MLP Categorical Actor class."""

    mlpCategoricalActor = MLPCategoricalActor(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=[hidden_sizes, hidden_sizes],
        activation=activation,
        weight_initialization_mode=weight_initialization_mode,
        shared=shared,
    )
    obs = torch.randn(obs_dim, dtype=torch.float32)

    out, logpro = mlpCategoricalActor(obs)
    assert isinstance(out, Categorical) and logpro is None, 'Failed!'
    out, logpro = mlpCategoricalActor(obs, torch.tensor(0, dtype=torch.float32))
    assert isinstance(out, Categorical) and isinstance(logpro, torch.Tensor), 'Failed!'

    dist = mlpCategoricalActor.dist(obs)
    assert isinstance(dist, Categorical), 'Failed'

    act, logpro = mlpCategoricalActor.predict(obs)
    assert isinstance(act, torch.Tensor) and isinstance(logpro, torch.Tensor), 'Failed!'


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

    if space_type == Discrete:
        action_space = space_type(act_dim)
    else:
        action_space = space_type(low=-1, high=1, shape=(act_dim,))

    actor_critic = ActorCritic(
        observation_space=observation_space,
        action_space=action_space,
        standardized_obs=standardized_obs,
        scale_rewards=scale_rewards,
        shared_weights=shared_weights,
        ac_kwargs=ac_kwargs,
        weight_initialization_mode=weight_initialization_mode,
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

    act = actor_critic.act(obs)
    assert isinstance(act, np.ndarray), 'Failed!'

    # TODO: Test anneal_exploration method.


if __name__ == '__main__':
    test_ActorCritic(100, 10, Discrete, False, False, False, 64, 'relu', 'kaiming_uniform')
