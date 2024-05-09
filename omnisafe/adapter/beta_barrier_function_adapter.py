# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Barrier Function Adapter with Beta Distribution for OmniSafe."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.envs.wrapper import AutoReset, CostNormalize, RewardNormalize, TimeLimit, Unsqueeze
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


# # pylint: disable-next=too-many-locals
def cbf(state: np.ndarray, eta: float = 0.99) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the Control Barrier Function (CBF) constraints.

    Args:
        state (np.ndarray | None): A numpy array containing the pendulum's current angular position
        (theta) and angular velocity (thetadot).
        eta (float): A scaling factor used to adjust the safety bounds.

    Returns:
        tuple containing two elements: 1. The minimum control torque that keeps the pendulum within
        the safety bounds. 2. The maximum control torque that keeps the pendulum within the safety
        bounds.

    Raises:
        ValueError: If the `eta` value is not within the open interval (0, 1).
    """
    g = 9.8
    m = 1
    length = 1
    tau = 5e-2
    theta_safety_bounds = [-1.0, 1.0]
    torque_bounds = [-15.0, 15.0]
    if (eta > 1 - 1e-3) or (eta < 1e-5):
        raise ValueError('eta should be inside (0, 1)')
    c1 = (3 * g) / (2 * length)
    c2 = 3 / (m * (length**2))

    theta, thetadot = state[0], state[1]
    theta_min, theta_max = theta_safety_bounds[0], theta_safety_bounds[1]
    thetadot_min, thetadot_max = -np.inf, np.inf
    u_min1 = (1 / c2) * (
        ((1 / (tau**2)) * (-eta * (theta - theta_min) - tau * thetadot)) - c1 * np.sin(theta)
    )
    u_max1 = (1 / c2) * (
        ((1 / (tau**2)) * (eta * (theta_max - theta) - tau * thetadot)) - c1 * np.sin(theta)
    )

    u_min2 = (1 / c2) * (((1 / (tau)) * (-eta * (thetadot - thetadot_min))) - c1 * np.sin(theta))
    u_max2 = (1 / c2) * (((1 / (tau)) * (eta * (thetadot_max - thetadot))) - c1 * np.sin(theta))

    u_min = max(u_min1, u_min2, torque_bounds[0])
    u_max = min(u_max1, u_max2, torque_bounds[1])

    return (u_min, u_max)


def vectorize_f(f: Callable) -> Callable:
    """Vectorize the function.

    Args:
        f (callable): A function that accepts 1D numpy arrays and returns a tuple (lower_bound, upper_bound).

    Returns:
        callable: A vectorized function that can process batches of torch tensors and return pairs of torch tensors.
    """

    def vectorized_f_(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Inner function to process the torch tensor batch.

        Args:
            obs (torch.Tensor): A batch of observations as torch tensors.

        Returns:
            tuple: Two torch tensors representing the lower and upper bounds for each observation in the batch.
        """
        obs = obs.cpu().detach().numpy()

        batch_size = obs.shape[0]
        lbs = torch.zeros([batch_size, 1])
        ubs = torch.zeros([batch_size, 1])
        for i in range(batch_size):
            lbs[i], ubs[i] = f(obs[i])

        lbs = torch.FloatTensor(lbs).reshape(batch_size, 1)
        ubs = torch.FloatTensor(ubs).reshape(batch_size, 1)

        return lbs, ubs

    return vectorized_f_


class BetaBarrierFunctionAdapter(OnPolicyAdapter):
    """Barrier Function Adapter with Beta Distribution for OmniSafe.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BetaBarrierFunctionAdapte`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.constraint_fn: Callable = vectorize_f(cbf)

    def _wrapper(
        self,
        obs_normalize: bool = False,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.

        .. warning::
            Since solving the optimization problem requires obtaining physical quantities with
            practical significance from state observations, the Beta Barrier Function Adapter does
            not support normalization of observations.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to False.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert not obs_normalize, 'Barrier function does not support observation normalization!'
        if self._env.need_time_limit_wrapper:
            assert (
                self._env.max_episode_steps
            ), 'You must define max_episode_steps as an integer\
                \nor cancel the use of the time_limit wrapper.'
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()
        obs, _ = self.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            with torch.no_grad():
                act, value_r, value_c, logp = agent.step(obs)
                lb, ub = self.constraint_fn(obs)
                final_act = lb + (ub - lb) * act

            next_obs, reward, cost, terminated, truncated, info = self.step(final_act)

            self._log_value(reward=reward, cost=cost, info=info)
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                obs[idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0
                        obs, _ = self.reset()
                    buffer.finish_path(last_value_r, last_value_c, idx)
