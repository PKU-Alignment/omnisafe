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
"""Test envs."""

from gymnasium.spaces import Box, Discrete

import helpers
from omnisafe.envs.core import make


@helpers.parametrize(
    num_envs=[1, 2],
)
def test_safety_gymnasium(num_envs) -> None:
    """Test envs."""
    env_id = 'SafetyPointGoal0-v0'
    env = make(env_id, num_envs=num_envs)

    obs_space = env.observation_space
    act_space = env.action_space

    assert isinstance(obs_space, Box)
    assert isinstance(act_space, Box)

    env.set_seed(0)
    obs, _ = env.reset()
    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
    else:
        assert obs.shape == (obs_space.shape[0],)

    act = env.sample_action()
    if num_envs > 1:
        act = act.repeat(num_envs, 1)

    obs, reward, cost, terminated, truncated, info = env.step(act)

    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
        assert reward.shape == (num_envs,)
        assert cost.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert isinstance(info, dict)
    else:
        assert obs.shape == (obs_space.shape[0],)
        assert reward.shape == ()
        assert cost.shape == ()
        assert terminated.shape == ()
        assert truncated.shape == ()
        assert isinstance(info, dict)

    env.close()


@helpers.parametrize(
    num_envs=[1],
)
def test_safety_gymnasium_modelbased(num_envs: int) -> None:
    """Test model-based envs."""
    env_id = 'SafetyPointGoal1-v0-modelbased'
    env = make(env_id, num_envs=num_envs, render_mode='rgb_array')
    if hasattr(env, 'coordinate_observation_space') or hasattr(env, 'observation_space'):
        obs_space = (
            env.coordinate_observation_space
            if env.coordinate_observation_space is not None
            else env.observation_space
        )
    else:
        raise NotImplementedError
    act_space = env.action_space

    assert isinstance(obs_space, Box)
    assert isinstance(act_space, Box)

    env.set_seed(0)
    obs, _ = env.reset()
    # env.render()
    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
    else:
        assert obs.shape == (obs_space.shape[0],)

    act = env.sample_action()
    if num_envs > 1:
        act = act.repeat(num_envs, 1)

    obs, reward, cost, terminated, truncated, info = env.step(act)

    assert obs.shape == (obs_space.shape[0],)
    assert reward.shape == ()
    assert cost.shape == ()
    if hasattr(env, 'get_cost_from_obs_tensor'):
        assert cost == env.get_cost_from_obs_tensor(obs.unsqueeze(0)).squeeze(0)
    assert terminated.shape == ()
    assert truncated.shape == ()
    assert isinstance(info, dict)

    env.close()


@helpers.parametrize(
    num_envs=[1],
    env_id=['Ant-v4', 'Hopper-v4', 'Walker2d-v4', 'Humanoid-v4', 'Swimmer-v4', 'HalfCheetah-v4'],
)
def test_mujoco(num_envs, env_id) -> None:
    """Test mujoco envs."""
    env = make(env_id, num_envs=num_envs)

    obs_space = env.observation_space

    act_space = env.action_space

    assert isinstance(obs_space, Box)
    assert isinstance(act_space, Box)

    env.set_seed(0)
    obs, _ = env.reset()
    # env.render()

    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
    else:
        assert obs.shape == (obs_space.shape[0],)

    act = env.sample_action()
    if num_envs > 1:
        act = act.repeat(num_envs, 1)

    obs, reward, cost, terminated, truncated, info = env.step(act)

    assert obs.shape == (obs_space.shape[0],)
    assert reward.shape == ()
    assert cost.shape == ()
    # assert cost == env.get_cost_from_obs_tensor(obs.unsqueeze(0)).squeeze(0)
    assert terminated.shape == ()
    assert truncated.shape == ()
    assert isinstance(info, dict)

    env.close()


@helpers.parametrize(
    num_envs=[1, 2],
)
def test_discrete(num_envs) -> None:
    """Test envs."""
    env_id = 'CartPole-v1'
    env = make(env_id, num_envs=num_envs)

    obs_space = env.observation_space
    act_space = env.action_space

    assert isinstance(obs_space, Box)
    assert isinstance(act_space, Discrete)

    env.set_seed(0)
    obs, _ = env.reset()
    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
    else:
        assert obs.shape == (obs_space.shape[0],)

    act = env.sample_action()

    obs, reward, cost, terminated, truncated, info = env.step(act)

    if num_envs > 1:
        assert obs.shape == (num_envs, obs_space.shape[0])
        assert reward.shape == (num_envs,)
        assert cost.shape == (num_envs,)
        assert terminated.shape == (num_envs,)
        assert truncated.shape == (num_envs,)
        assert isinstance(info, dict)
    else:
        assert obs.shape == (obs_space.shape[0],)
        assert reward.shape == ()
        assert cost.shape == ()
        assert terminated.shape == ()
        assert truncated.shape == ()
        assert isinstance(info, dict)

    env.close()


test_discrete(num_envs=2)
