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
"""BarrierFunction Adapter for OmniSafe."""

from __future__ import annotations

import torch
import numpy as np
from rich.progress import track

from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.common.barrier_solver import PendulumSolver
from omnisafe.common.barrier_comp import BarrierCompensator

from omnisafe.envs.wrapper import (
    AutoReset,
    CostNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)


def cbf(state=None, eta: float = 0.99):
    """
    Calculates CBF constraint set at a given state. Default is
    the current state.
    """

    state = state
    g = 9.8
    m = 1
    l = 1
    tau = 5e-2
    theta_safety_bounds = [-1.0, 1.0]
    thetadot_safety_bounds = [-np.inf, np.inf]
    torque_bounds = [-15.0, 15.0]
    if (eta>1-1e-3) or (eta<1e-5):
        raise ValueError("eta should be inside (0, 1)")
    c1 = ((3 * g)/(2 * l))
    c2 = (3 /(m * (l ** 2)))

    theta, thetadot = state[0], state[1]
    theta_min, theta_max = theta_safety_bounds[0], theta_safety_bounds[1]
    thetadot_min, thetadot_max = thetadot_safety_bounds[0], thetadot_safety_bounds[1]
    u_min1 = (1/c2) * (((1 / (tau **2)) * (-eta * (theta - theta_min) - tau * thetadot)) - c1 * np.sin(theta) )
    u_max1 = (1/c2) * (((1 / (tau **2)) * ( eta * (theta_max - theta) - tau * thetadot)) - c1 * np.sin(theta) )

    
    u_min2 = (1/c2) * (((1 / (tau)) * (-eta * (thetadot - thetadot_min))) - c1 * np.sin(theta) )
    u_max2 = (1/c2) * (((1 / (tau)) * ( eta * (thetadot_max - thetadot))) - c1 * np.sin(theta) )

    u_min = max(u_min1, u_min2, torque_bounds[0])
    u_max = min(u_max1, u_max2, torque_bounds[1])
    
    u_min=torque_bounds[0]
    u_max=torque_bounds[1]
    if u_min>u_max:
        raise ValueError("Infeasible")
    else:
        return [u_min, u_max]

def vectorize_f(f): #--vipul :added action_dim
    """
    Converts a function f defined on 1D numpy arrays and outputting pairs of
    scalars into a vectorized function accepting batches of
    torch tensorized arrays and output pairs of torch tensors.
    """

    def vectorized_f_(obs): #--vipul :added action_dim

        obs = obs.cpu().detach().numpy()

        if len(obs.shape) == 1:  # check to see if obs is a batch or single obs
            batch_size = 1
            lbs, ubs = f(obs)
            lbs=np.array(lbs)
            ubs=np.array(ubs)
            #lbs = -5
            #ubs = 5

        else:
            batch_size = obs.shape[0]
            lbs = np.zeros([batch_size, 1])
            ubs = np.zeros([batch_size, 1])
            for i in range(batch_size):
                lbs[i], ubs[i] = f(obs[i])

        lbs = torch.FloatTensor(lbs).reshape(batch_size, 1)
        ubs = torch.FloatTensor(ubs).reshape(batch_size, 1)
        
        return lbs, ubs

    return vectorized_f_


class BetaBarrierFunctionAdapter(OnPolicyAdapter):
    """BarrierFunction Adapter for OmniSafe.

    The BarrierFunction Adapter is used to establish the logic of interaction between agents and the 
    environment based on control barrier functions. Its key feature is the introduction of action 
    compensators and barrier function solvers.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of parallel environments.
        seed (int): The random seed.
        cfgs (Config): The configuration passed from yaml file.
    """

    def __init__(self, env_id: str, num_envs: int, seed: int, cfgs: Config) -> None:
        """Initialize an instance of :class:`BarrierFunctionAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self.solver = None
        self.compensator = None
        self.first_iter = 1
        self.constraint_fn = vectorize_f(cbf)

    def _wrapper(
        self,
        obs_normalize: bool = False,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
    ) -> None:
        """Wrapper the environment.
        
        .. warning::
            Since solving the optimization problem requires obtaining physical quantities with practical 
            significance from state observations, the Barrier Function Adapter does not support 
            normalization of observations.

        Args:
            obs_normalize (bool, optional): Whether to normalize the observation. Defaults to False.
            reward_normalize (bool, optional): Whether to normalize the reward. Defaults to True.
            cost_normalize (bool, optional): Whether to normalize the cost. Defaults to True.
        """
        assert not obs_normalize, 'Barrier function does not support observation normalization!'
        if reward_normalize:
            self._env = RewardNormalize(self._env, device=self._device)
        if cost_normalize:
            self._env = CostNormalize(self._env, device=self._device)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)
        self._eval_env = Unsqueeze(self._eval_env, device=self._device)

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()
        obs, _ = self.reset()
        while abs(self._env.unwrapped.state[0]) > 1:
            obs, _ = self._env.reset()
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            with torch.no_grad():
                act, value_r, value_c, logp = agent.step(obs)
                lb, ub = self.constraint_fn(obs)
                final_act = lb + (ub-lb)*act

            next_obs, reward, cost, terminated, truncated, info = self.step(final_act)
            
            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})
            logger.store({'Metrics/angle': info.get('original_cost', cost).cpu()})

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
                        while abs(self._env.unwrapped.state[0]) > 1:
                            obs, _ = self._env.reset()
                    buffer.finish_path(last_value_r, last_value_c, idx)
        self.first_iter = 0

