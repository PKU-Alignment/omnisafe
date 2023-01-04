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
"""Implementation of the CAP algorithm."""

import numpy as np

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.planner import CCEPlanner
from omnisafe.algorithms.model_based.policy_gradient import PolicyGradientModelBased
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils.config_utils import namedtuple2dict


@registry.register
class CAP(
    PolicyGradientModelBased, CCEPlanner, Lagrange
):  # pylint: disable=too-many-instance-attributes
    """The Conservative and Adaptive Penalty (CAP) algorithm.

    References:
        Title: Conservative and Adaptive Penalty for Model-Based Safe Reinforcement Learning
        Authors: Yecheng Jason Ma, Andrew Shen, Osbert Bastani, Dinesh Jayaraman.
        URL: https://arxiv.org/abs/2112.07701
    """

    def __init__(self, env_id, cfgs) -> None:
        PolicyGradientModelBased.__init__(
            self,
            env_id=env_id,
            cfgs=cfgs,
        )
        Lagrange.__init__(self, **namedtuple2dict(self.cfgs.lagrange_cfgs), device=self.cfgs.device)
        CCEPlanner.__init__(
            self,
            algo=self.algo,
            cfgs=self.cfgs,
            device=self.device,
            env=self.env,
            models=self.virtual_env,
            **namedtuple2dict(self.cfgs.mpc_config),
            lagrangian_multiplier=self.lagrangian_multiplier,
        )
        # Set up model saving
        what_to_save = {
            'dynamics': self.dynamics,
        }
        self.logger.setup_torch_saver(what_to_save=what_to_save)
        self.logger.torch_save()

    def algorithm_specific_logs(self, time_step):
        """Log algo parameter"""
        super().algorithm_specific_logs(time_step)
        self.logger.log_tabular('Loss/DynamicsTrainMseLoss')
        self.logger.log_tabular('Loss/DynamicsValMseLoss')
        self.logger.log_tabular(
            'Penalty', self.lambda_range_projection(self.lagrangian_multiplier).item()
        )

    def update_dynamics_model(self):
        """Update dynamics."""
        state = self.off_replay_buffer.obs_buf[: self.off_replay_buffer.size, :]
        action = self.off_replay_buffer.act_buf[: self.off_replay_buffer.size, :]
        reward = self.off_replay_buffer.rew_buf[: self.off_replay_buffer.size]
        cost = self.off_replay_buffer.cost_buf[: self.off_replay_buffer.size]
        next_state = self.off_replay_buffer.obs_next_buf[: self.off_replay_buffer.size, :]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        if self.env.env_type == 'mujoco-velocity':
            labels = np.concatenate(
                (
                    np.reshape(reward, (reward.shape[0], -1)),
                    np.reshape(cost, (cost.shape[0], -1)),
                    delta_state,
                ),
                axis=-1,
            )
        elif self.env.env_type == 'gym':
            labels = np.concatenate(
                (np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1
            )
        train_mse_losses, val_mse_losses = self.dynamics.train(
            inputs, labels, batch_size=256, holdout_ratio=0.2
        )

        self.logger.store(
            **{
                'Loss/DynamicsTrainMseLoss': train_mse_losses,
                'Loss/DynamicsValMseLoss': val_mse_losses,
            }
        )

    def select_action(self, time_step, state, env):
        """action selection"""
        action = self.get_action(np.array(state))
        return action, None

    def store_real_data(
        self,
        time_step,
        ep_len,
        state,
        action_info,
        action,
        reward,
        cost,
        terminated,
        truncated,
        next_state,
        info,
    ):  # pylint: disable=too-many-arguments
        """store real data"""
        if not terminated and not truncated and not info['goal_met']:
            # Current goal position is not related to the last goal position, so do not store.
            self.off_replay_buffer.store(
                obs=state, act=action, rew=reward, cost=cost, next_obs=next_state, done=truncated
            )

    def algo_reset(self):
        """reset planner"""
        ep_costs = self.logger.get_stats('Metrics/EpCost')[0]
        # update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        self.planner_reset(self.lagrangian_multiplier)

    def set_algorithm_specific_actor_critic(self):
        """Initialize Soft Actor-Critic"""
