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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config


class OnPolicyAdapter(OnlineAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._reset_log()

        self._debug_logged = False

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
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            '''调试
            if torch.any(torch.isnan(reward)):
                print(f"[DEBUG Rollout] Step {step}: NaN reward from env.step(): {reward}")
                print(f"[DEBUG Rollout] Info keys: {list(info.keys())}")
                if 'original_reward' in info:
                    print(f"[DEBUG Rollout] Original reward in info: {info['original_reward']}")
            
            if torch.any(torch.isnan(cost)):
                print(f"[DEBUG Rollout] Step {step}: NaN cost from env.step(): {cost}")
                if 'original_cost' in info:
                    print(f"[DEBUG Rollout] Original cost in info: {info['original_cost']}")
            '''

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
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
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out or epoch_end:  #here i add epoch_end to justify the log
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _log_value(
        self,
        reward: torch.Tensor,
        cost: torch.Tensor,
        info: dict[str, Any],
    ) -> None:
        """Log value.

        .. note::
            OmniSafe uses :class:`RewardNormalizer` wrapper, so the original reward and cost will
            be stored in ``info['original_reward']`` and ``info['original_cost']``.

        Args:
            reward (torch.Tensor): The immediate step reward.
            cost (torch.Tensor): The immediate step cost.
            info (dict[str, Any]): Some information logged by the environment.
        """

        '''
        if hasattr(self, '_debug_logged') and not self._debug_logged:
            print(f"[DEBUG _log_value] First call debug:")
            print(f"  reward shape: {reward.shape}, value: {reward}")
            print(f"  cost shape: {cost.shape}, value: {cost}")
            print(f"  info keys: {list(info.keys())}")
            if 'original_reward' in info:
                print(f"  original_reward: {info['original_reward']}")
            if 'original_cost' in info:
                print(f"  original_cost: {info['original_cost']}")
            print(f"  _ep_ret before: {self._ep_ret}")
            print(f"  _ep_cost before: {self._ep_cost}")
            print(f"  _ep_len before: {self._ep_len}")
            self._debug_logged = True
        '''

        #FIX BEGIN
        raw_reward = info.get('original_reward', reward)
        raw_cost = info.get('original_cost', cost)
        
        if torch.any(torch.isnan(raw_reward)):
            #print(f"[CRITICAL _log_value] NaN raw_reward detected! raw_reward={raw_reward}, reward={reward}")
            #if 'original_reward' in info:
                #print(f"  original_reward in info: {info['original_reward']}")
            raw_reward = torch.nan_to_num(raw_reward, nan=0.0)
        
        if torch.any(torch.isnan(raw_cost)):
            #print(f"[CRITICAL _log_value] NaN raw_cost detected! raw_cost={raw_cost}, cost={cost}")
            #if 'original_cost' in info:
                #print(f"  original_cost in info: {info['original_cost']}")
            raw_cost = torch.nan_to_num(raw_cost, nan=0.0)
    
        if torch.any(torch.isnan(self._ep_ret)):
            #print(f"[CRITICAL _log_value] _ep_ret is NaN before addition! Value: {self._ep_ret}")
            self._ep_ret = torch.zeros_like(self._ep_ret)
        
        if torch.any(torch.isnan(self._ep_cost)):
            #print(f"[CRITICAL _log_value] _ep_cost is NaN before addition! Value: {self._ep_cost}")
            self._ep_cost = torch.zeros_like(self._ep_cost)
        
        if torch.any(torch.isnan(self._ep_len)):
            #print(f"[CRITICAL _log_value] _ep_len is NaN before addition! Value: {self._ep_len}")
            self._ep_len = torch.zeros_like(self._ep_len)
        
        self._ep_ret += raw_reward.cpu()
        self._ep_cost += raw_cost.cpu()
        self._ep_len += 1
        
        '''
        if torch.any(torch.isnan(self._ep_ret)):
            print(f"[CRITICAL _log_value] _ep_ret became NaN after addition!")
        if torch.any(torch.isnan(self._ep_cost)):
            print(f"[CRITICAL _log_value] _ep_cost became NaN after addition!")
        if torch.any(torch.isnan(self._ep_len)):
            print(f"[CRITICAL _log_value] _ep_len became NaN after addition!")
        '''

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``."""
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        
        '''
        print(f"[DEBUG _log_metrics] Called for idx={idx}")
        print(f"  _ep_ret: {self._ep_ret}, type: {type(self._ep_ret)}")
        print(f"  _ep_cost: {self._ep_cost}, type: {type(self._ep_cost)}")
        print(f"  _ep_len: {self._ep_len}, type: {type(self._ep_len)}")
        
        if torch.any(torch.isnan(self._ep_ret)):
            print(f"[ERROR _log_metrics] _ep_ret contains NaN! Values: {self._ep_ret}")
        if torch.any(torch.isnan(self._ep_cost)):
            print(f"[ERROR _log_metrics] _ep_cost contains NaN! Values: {self._ep_cost}")
        if torch.any(torch.isnan(self._ep_len)):
            print(f"[ERROR _log_metrics] _ep_len contains NaN! Values: {self._ep_len}")
        '''
        
        ep_ret_val = self._ep_ret[idx]
        ep_cost_val = self._ep_cost[idx]
        ep_len_val = self._ep_len[idx]
        
        #print(f"  ep_ret_val: {ep_ret_val}, type: {type(ep_ret_val)}")
        #print(f"  ep_cost_val: {ep_cost_val}, type: {type(ep_cost_val)}")
        #print(f"  ep_len_val: {ep_len_val}, type: {type(ep_len_val)}")
        
        if torch.isnan(ep_ret_val) or torch.isinf(ep_ret_val):
            #print(f"[FIXING _log_metrics] EpRet[{idx}] = {ep_ret_val}, setting to 0.0")
            ep_ret_val = torch.tensor(0.0, dtype=torch.float32)
        
        if torch.isnan(ep_cost_val) or torch.isinf(ep_cost_val):
            #print(f"[FIXING _log_metrics] EpCost[{idx}] = {ep_cost_val}, setting to 0.0")
            ep_cost_val = torch.tensor(0.0, dtype=torch.float32)
        
        if torch.isnan(ep_len_val) or torch.isinf(ep_len_val):
            #print(f"[FIXING _log_metrics] EpLen[{idx}] = {ep_len_val}, setting to 0.0")
            ep_len_val = torch.tensor(0.0, dtype=torch.float32)
        
        logger.store(
            {
                'Metrics/EpRet': ep_ret_val,
                'Metrics/EpCost': ep_cost_val,
                'Metrics/EpLen': ep_len_val,
            },
        )
    
        #print(f"[DEBUG _log_metrics] Stored values - EpRet: {ep_ret_val}, EpCost: {ep_cost_val}, EpLen: {ep_len_val}")

    def _reset_log(self, idx: int | None = None) -> None:
        """Reset the episode return, episode cost and episode length.

        Args:
            idx (int or None, optional): The index of the environment. Defaults to None
                (single environment).
        """
        if idx is None:
            self._ep_ret = torch.zeros(self._env.num_envs)
            self._ep_cost = torch.zeros(self._env.num_envs)
            self._ep_len = torch.zeros(self._env.num_envs)
        else:
            self._ep_ret[idx] = 0.0
            self._ep_cost[idx] = 0.0
            self._ep_len[idx] = 0.0

        if torch.any(torch.isnan(self._ep_ret)):
            #print(f"[ERROR _reset_log] _ep_ret initialized as NaN!")
            self._ep_ret = torch.zeros_like(self._ep_ret)
        
        if torch.any(torch.isnan(self._ep_cost)):
            #print(f"[ERROR _reset_log] _ep_cost initialized as NaN!")
            self._ep_cost = torch.zeros_like(self._ep_cost)
        
        if torch.any(torch.isnan(self._ep_len)):
            #print(f"[ERROR _reset_log] _ep_len initialized as NaN!")
            self._ep_len = torch.zeros_like(self._ep_len)
