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

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.model_based.policy_gradient import PolicyGradientModelBased


@registry.register
class SafeLoop(PolicyGradientModelBased):
    """safeloop"""

    def __init__(self, algo='safeloop', clip=0.2, **cfgs):
        super().__init__(algo=algo, **cfgs)
        self.clip = clip
        self.device = torch.device(self.cfgs['device'])

    def algorithm_specific_logs(self, timestep):
        """log algo parameter"""
        super().algorithm_specific_logs(timestep)
        if timestep >= self.start_timesteps:
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

    def update(self):
        """TODO"""

    # Set up model saving
    def update_actor_critic(self, data):
        """update ac"""
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, _ = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        log_pi = pi_info['LogPi']
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi.to(self.device) + self.target_entropy)).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.actor_critic.parameters(), self.actor_critic_targ.parameters()
            ):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
        self.logger.store(
            **{
                'Loss/Pi': loss_pi.item(),
                'Loss/alpha': alpha_loss.item(),
                'Loss/Q-networks': loss_q.item(),
            }
        )

    def compute_loss_q(self, data):
        """Set up function for computing SAC Q-losses"""
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q1 = self.actor_critic.q1(o, a)
        q2 = self.actor_critic.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critic.pi(o2)
            # Target Q-values
            q1_pi_targ = self.actor_critic_targ.q1(o2, a2)
            q2_pi_targ = self.actor_critic_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.cpu().detach().numpy(), Q2Vals=q2.cpu().detach().numpy())

        return loss_q, q_info

    def compute_loss_pi(self, data):
        """compute loss of pi"""
        o = data['obs']
        a = data['act']
        pi, logp_pi = self.actor_critic.pi(o)
        q1_pi = self.actor_critic.q1(o, pi)
        q2_pi = self.actor_critic.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        if self.use_bc_loss:
            # BC loss inspired from TD3_BC offline RL algorithm
            # Refer https://github.com/sfujim/TD3_BC
            lmbda = 2.5 / q_pi.abs().mean().detach()
            loss_pi = (self.alpha * logp_pi - q_pi).mean() * lmbda + F.mse_loss(pi, a)
        else:
            # Entropy-regularized policy loss
            loss_pi = (self.alpha * logp_pi - q_pi).mean()

        pi_info = dict(LogPi=logp_pi.cpu().detach())

        return loss_pi, pi_info

    def update_dynamics_model(self):
        """update dynamics"""
        state = self.replay_buffer.state[: self.replay_buffer.size, :]
        action = self.replay_buffer.action[: self.replay_buffer.size, :]
        reward = self.replay_buffer.reward[: self.replay_buffer.size]
        next_state = self.replay_buffer.next_state[: self.replay_buffer.size, :]
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
