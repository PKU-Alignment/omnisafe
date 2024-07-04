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
"""Implementation of the PPO algorithm with Control Barrier Function and Beta Actor."""

from __future__ import annotations

import torch

from omnisafe.adapter.beta_barrier_function_adapter import BetaBarrierFunctionAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed


@registry.register
class PPOBetaCBF(PPO):
    """The PPO algorithm with CBF and Beta Actor.

    References:
        - Title: Sampling-based Safe Reinforcement Learning for Nonlinear Dynamical Systems
        - Authors: Wesley A. Suttle, Vipul K. Sharma, Krishna C. Kosaraju, S. Sivaranjani, Ji Liu,
            Vijay Gupta, Brian M. Sadler.
        - URL: `PPOBetaCBF <https://proceedings.mlr.press/v238/suttle24a/suttle24a.pdf>`_
    """

    def _init_env(self) -> None:
        self._env: BetaBarrierFunctionAdapter = BetaBarrierFunctionAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        This section of the logic is consistent with PPO, except that it does not record the
        standard deviation of the actor distribution.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * distribution.entropy().mean()
        # useful extra info
        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Loss/Loss_pi': loss.mean().item(),
            },
        )
        return loss
