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
"""OffPolicy Adapter for OmniSafe."""

from functools import partial
from typing import Dict, Optional

import torch
from gymnasium import spaces

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.config import Config
from omnisafe.envs.wrapper import (
    ActionScale,
    ActionRepeat,
    AutoReset,
    CostNormalize,
    ObsNormalize,
    RewardNormalize,
    TimeLimit,
    Unsqueeze,
)
import time
from omnisafe.envs.core import make, support_envs

from typing import Callable, Union

class ModelBasedAdapter(OnlineAdapter):
    """OffPolicy Adapter for OmniSafe."""

    def __init__(  # pylint: disable=too-many-arguments
        self, env_id: str, num_envs: int, seed: int, cfgs: Config
    ) -> None:
        assert env_id in support_envs(), f'Env {env_id} is not supported.'

        self._env_id = env_id
        self._env = make(env_id, num_envs=num_envs)
        self._wrapper(
            obs_normalize=cfgs.algo_cfgs.obs_normalize,
            reward_normalize=cfgs.algo_cfgs.reward_normalize,
            cost_normalize=cfgs.algo_cfgs.cost_normalize,
            action_repeat=cfgs.algo_cfgs.action_repeat,

        )
        self._env.set_seed(seed)
        self._cfgs = cfgs
        self._device = cfgs.train_cfgs.device

        self._ep_ret: torch.Tensor
        self._ep_cost: torch.Tensor
        self._ep_len: torch.Tensor
        self._current_obs, _ = self.reset()
        self._max_ep_len = 1000
        self._reset_log()
        self._last_dynamics_update = 0
        self._last_policy_update = 0
    def _wrapper(
        self,
        obs_normalize: bool = True,
        reward_normalize: bool = True,
        cost_normalize: bool = True,
        action_repeat: int = 1,
    ):
        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, time_limit=1000)
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env)
        if obs_normalize:
            self._env = ObsNormalize(self._env)
        if reward_normalize:
            self._env = RewardNormalize(self._env)
        if cost_normalize:
            self._env = CostNormalize(self._env)
        self._env = ActionScale(self._env, low=-1.0, high=1.0)
        self._env = ActionRepeat(self._env, times=action_repeat)

        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env)

    def roll_out(
            self,
            current_step: int,
            roll_out_step: int,
            use_actor_critic: bool,
            act_fn: Callable,
            store_data_func: Callable,
            update_dynamics_model: Callable,
            logger: Logger,
            algo_reset_func: Union[Callable, None]=None,
            update_actor_critic: Union[Callable, None]=None,
        ) -> int:
        epoch_start_time = time.time()

        update_actor_critic_time = 0
        update_dynamics_time = 0
        epoch_steps = 0

        while epoch_steps < roll_out_step:
            action, action_info = act_fn(current_step, self._current_obs)
            next_state, reward, cost, terminated, truncated, info = self.step(action)
            epoch_steps += info['num_step']
            current_step += info['num_step']
            self._log_value(reward=reward, cost=cost, info=info)

            store_data = {
                'current_step': current_step,
                'ep_len': self._ep_len,
                'current_obs': self._current_obs,
                'action_info': action_info,
                'action': action,
                'reward': reward,
                'cost': cost,
                'terminated': terminated,
                'truncated': truncated,
                'next_state': next_state,
                'info': info,
            }
            store_data_func(
                current_step,
                self._ep_len,
                self._current_obs,
                action,
                reward,
                cost,
                terminated,
                truncated,
                next_state,
                info,
                action_info,
            )
            self._current_obs = next_state
            if terminated or truncated:
                self._log_metrics(logger)
                self._reset_log()
                self._current_obs, _ = self.reset()
                if algo_reset_func is not None:
                    algo_reset_func(current_step)
            if (
                current_step % self._cfgs.algo_cfgs.update_dynamics_cycle < self._cfgs.algo_cfgs.action_repeat
                and current_step - self._last_dynamics_update >= self._cfgs.algo_cfgs.update_dynamics_cycle
            ):
                update_dynamics_start = time.time()
                update_dynamics_model(current_step)
                self._last_dynamics_update = current_step
                update_dynamics_time += time.time() - update_dynamics_start

            if (
                use_actor_critic
                and current_step % self._cfgs.algo_cfgs.update_policy_cycle < self._cfgs.algo_cfgs.action_repeat
                and current_step - self._last_policy_update >= self._cfgs.algo_cfgs.update_policy_cycle
            ):
                update_actor_critic_start = time.time()
                update_actor_critic(current_step)
                self._last_policy_update = current_step
                update_actor_critic_time += time.time() - update_actor_critic_start

        epoch_time = time.time() - epoch_start_time
        logger.store(**{'Time/Epoch': epoch_time})
        logger.store(**{'Time/UpdateDynamics': update_dynamics_time})
        roll_out_time = epoch_time - update_dynamics_time
        if use_actor_critic:
            logger.store(**{'Time/UpdateActorCritic': update_actor_critic_time})
            roll_out_time -= update_actor_critic_time
        logger.store(**{'Time/Rollout': roll_out_time})
        return current_step


    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: Dict,
        **kwargs,  # pylint: disable=unused-argument
    ) -> None:
        """Log value."""
        self._ep_ret += info.get('original_reward', reward).cpu()
        self._ep_cost += info.get('original_cost', cost).cpu()
        self._ep_len += info.get('step_num', 1)

    def _log_metrics(self, logger: Logger) -> None:
        """Log metrics."""
        logger.store(
            **{
                'Metrics/EpRet': self._ep_ret,
                'Metrics/EpCost': self._ep_cost,
                'Metrics/EpLen': self._ep_len,
            }
        )

    def _reset_log(self, idx: Optional[int] = None) -> None:
        """Reset log."""
        self._ep_ret = torch.zeros(1)
        self._ep_cost = torch.zeros(1)
        self._ep_len = torch.zeros(1)

    def check_violation(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[1] == self.observation_space.shape[0]
        if self._env_id == "Ant-v4":
            min_z, max_z = 0.2, 1.0
            is_finite = torch.isfinite(obs).all()
            is_between = torch.logical_and(min_z < obs[:, 0], obs[:, 0] < max_z)
            is_healthy = torch.logical_and(is_finite, is_between)
        elif self._env_id == "Humanoid-v4":
            min_z, max_z = 1.0, 2.0
            is_healthy = torch.logical_and(min_z < obs[:, 0],  obs[:, 0] < max_z)
        elif self._env_id == "Hopper-v4":
            z, angle = obs[:, 0:2]
            state = obs[:, 1:]
            min_state, max_state = -100.0, 100.0
            min_z, max_z = (0.7, float("inf"))
            min_angle, max_angle = (-0.2, 0.2)
            healthy_state = torch.logical_and(min_state < state, state < max_state)
            healthy_z = torch.logical_and(min_z < z, z < max_z)
            healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
            is_healthy = torch.all(torch.stack([healthy_state, healthy_z, healthy_angle]), dim=0)
        elif self._env_id == "walker2d-v4":
            z, angle = obs[0:2]
            min_z, max_z = (0.8, 2)
            min_angle, max_angle = (-1, 1)
            healthy_z = torch.logical_and(min_z < z, z < max_z)
            healthy_angle = torch.logical_and(min_angle < angle, angle < max_angle)
            is_healthy = torch.logical_and(healthy_z, healthy_angle)

        assert is_healthy.shape == obs.shape[:1], f"{is_healthy.shape} != {obs.shape[:1]}"

        return is_healthy
