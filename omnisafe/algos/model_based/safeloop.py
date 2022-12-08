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
"""Safe-Loop"""

import itertools
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from omnisafe.algos import registry
from omnisafe.algos.model_based.models.actor_critic import SoftActorCritic
from omnisafe.algos.model_based.planner import Planner
from omnisafe.algos.model_based.policy_gradient import PolicyGradientModelBased


@registry.register
class SafeLoop(PolicyGradientModelBased, Planner):  # pylint: disable=too-many-instance-attributes
    """SafeLoop"""

    def __init__(self, algo='safe-loop', clip=0.2, **cfgs):
        PolicyGradientModelBased.__init__(self, algo=algo, **cfgs)
        Planner.__init__(
            self,
            self.device,
            self.env,
            self.virtual_env,
            self.actor_critic,
            **self.cfgs['mpc_config'],
        )
        self.clip = clip
        if self.cfgs['automatic_alpha_tuning']:
            self.target_entropy = -torch.prod(
                torch.Tensor(self.env.action_space.shape).to(self.device)
            ).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=self.cfgs['sac_lr'])
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.cfgs['alpha_init']

    def set_algorithm_specific_actor_critic(self):
        """Initialize Soft Actor-Critic"""
        self.actor_critic = SoftActorCritic(
            self.env.ac_state_size,
            self.env.action_space,
            **dict(hidden_sizes=self.cfgs['ac_hidden_sizes']),
        ).to(self.device)
        self.actor_critic_targ = deepcopy(self.actor_critic)
        # Freeze target networks with respect to optimizer (only update via polyak averaging)
        for ac_targ_parameters in self.actor_critic_targ.parameters():
            ac_targ_parameters.requires_grad = False
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(
            self.actor_critic.q1.parameters(), self.actor_critic.q2.parameters()
        )
        # Set up optimizer for policy and q-function
        self.pi_optimizer = Adam(self.actor_critic.pi.parameters(), lr=self.cfgs['sac_lr'])
        self.q_optimizer = Adam(self.q_params, lr=self.cfgs['sac_lr'])
        self.v_optimizer = Adam(self.actor_critic.v.parameters(), lr=self.cfgs['sac_lr'])
        return self.actor_critic

    def algorithm_specific_logs(self, timestep):
        """Log algo parameter"""
        super().algorithm_specific_logs(timestep)
        if timestep >= self.cfgs['update_policy_start_timesteps']:
            self.logger.log_tabular('Loss/Pi')
            self.logger.log_tabular('Loss/alpha')
            self.logger.log_tabular('Loss/Q-networks')
            self.logger.log_tabular('Loss/DynamicsTrainLoss')
            self.logger.log_tabular('Loss/DynamicsValLoss')
        else:
            self.logger.store(
                **{
                    'Loss/Pi': 0,
                    'Loss/alpha': 0,
                    'Loss/Q-networks': 0,
                }
            )
            self.logger.log_tabular('Loss/Pi')
            self.logger.log_tabular('Loss/alpha')
            self.logger.log_tabular('Loss/Q-networks')
            self.logger.log_tabular('Loss/DynamicsTrainLoss')
            self.logger.log_tabular('Loss/DynamicsValLoss')

    def update_actor_critic(self, timestep):
        """update actor and critic"""
        if timestep >= self.cfgs['update_policy_start_timesteps']:
            for _ in range(self.cfgs['update_policy_iters']):
                # Get one batch data from Off-policy buffer
                data = self.off_replay_buffer.sample_batch()
                # Update critic
                self.update_value_net(data)
                # Freeze Critic
                self.freeze_critic_network(requires_grad=False)
                # Update Actor
                log_pi = self.update_policy_net(data)
                # Update Alpha
                if self.cfgs['automatic_alpha_tuning']:
                    self.update_alpha(log_pi)
                # Update target Critic
                self.update_target_critic()
                # Unfreeze Critic
                self.freeze_critic_network(requires_grad=True)

    def freeze_critic_network(self, requires_grad=True):
        """Freeze Q-networks for saving computing resources."""
        for parameter in self.q_params:
            parameter.requires_grad = requires_grad

    def update_value_net(self, data):
        """Value function learning"""
        # Run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, _ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()
        self.logger.store(**{'Loss/Q-networks': loss_q.item()})

    def update_policy_net(self, data):
        """Update policy"""
        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        log_pi = pi_info['LogPi']
        loss_pi.backward()
        self.pi_optimizer.step()
        self.logger.store(**{'Loss/Pi': loss_pi.item()})
        return log_pi

    def update_alpha(self, log_pi):
        """Update"""
        alpha_loss = -(self.log_alpha * (log_pi.to(self.device) + self.target_entropy)).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()
        self.logger.store(
            **{
                'Loss/alpha': alpha_loss.item(),
            }
        )

    def update_target_critic(self):
        """Update target networks by polyak averaging."""
        with torch.no_grad():
            for parameter, parameter_target in zip(
                self.actor_critic.parameters(), self.actor_critic_targ.parameters()
            ):
                parameter_target.data.mul_(self.cfgs['polyak'])
                parameter_target.data.add_((1 - self.cfgs['polyak']) * parameter.data)

    def compute_loss_q(self, data):  # pylint: disable=too-many-locals
        """Set up function for computing SAC Q-losses"""
        state, action, reward, next_state, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs2'],
            data['done'],
        )
        q_value1 = self.actor_critic.q1(state, action)
        q_value2 = self.actor_critic.q2(state, action)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            next_action, logp_a2 = self.actor_critic.pi(next_state)
            # Target Q-values
            q1_pi_targ = self.actor_critic_targ.q1(next_state, next_action)
            q2_pi_targ = self.actor_critic_targ.q2(next_state, next_action)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = reward + self.cfgs['gamma'] * (1 - done) * (q_pi_targ - self.alpha * logp_a2)

        # Bellman backup
        loss_q1 = ((q_value1 - backup) ** 2).mean()
        loss_q2 = ((q_value2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q_value1.cpu().detach().numpy(), Q2Vals=q_value2.cpu().detach().numpy()
        )

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """compute loss of pi"""
        state = data['obs']
        action = data['act']
        pi_action, logp_pi = self.actor_critic.pi(state)
        q1_pi = self.actor_critic.q1(state, pi_action)
        q2_pi = self.actor_critic.q2(state, pi_action)
        q_pi = torch.min(q1_pi, q2_pi)
        if self.cfgs['use_bc_loss']:
            # BC loss inspired from TD3_BC offline RL algorithm
            # Refer https://github.com/sfujim/TD3_BC
            lmbda = 2.5 / q_pi.abs().mean().detach()
            loss_pi = (self.alpha * logp_pi - q_pi).mean() * lmbda + F.mse_loss(pi_action, action)
        else:
            # Entropy-regularized policy loss
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.cpu().detach())

        return loss_pi, pi_info

    def update_dynamics_model(self):
        """Update dynamics."""
        state = self.off_replay_buffer.obs_buf[: self.off_replay_buffer.size, :]
        action = self.off_replay_buffer.act_buf[: self.off_replay_buffer.size, :]
        reward = self.off_replay_buffer.rew_buf[: self.off_replay_buffer.size]
        next_state = self.off_replay_buffer.obs2_buf[: self.off_replay_buffer.size, :]
        delta_state = next_state - state
        inputs = np.concatenate((state, action), axis=-1)
        labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
        trainloss, valloss = self.dynamics.train(inputs, labels, batch_size=256, holdout_ratio=0.2)
        self.logger.store(
            **{
                'Loss/DynamicsTrainLoss': trainloss,
                'Loss/DynamicsValLoss': valloss,
            }
        )

    def select_action(self, timestep, state, env):
        """action selection"""
        if timestep < self.cfgs['update_policy_start_timesteps']:
            action = self.env.action_space.sample()
        else:
            action = self.get_action(np.array(state), env=env)
            action = action + np.random.normal(action.shape) * self.cfgs['exploration_noise']
        action = np.clip(action, env.action_space.low, env.action_space.high)
        return action, None

    def store_real_data(
        self,
        timestep,
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
        self.planner_reset()
