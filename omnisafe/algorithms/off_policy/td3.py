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
"""Implementation of the DDPG algorithm."""


import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG

@registry.register
class TD3(DDPG):  # pylint: disable=too-many-instance-attributes
    """Continuous control with deep reinforcement learning (DDPG) Algorithm.

    References:
        Paper Name: Continuous control with deep reinforcement learning.
        Paper author: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess,
                      Tom Erez, Yuval Tassa, David Silver, Daan Wierstra.
        Paper URL: https://arxiv.org/abs/1509.02971

    """

    def __init__(
        self,
        env_id: str,
        cfgs=None,
        algo: str = 'TD3',
        wrapper_type: str = 'OffPolicyEnvWrapper',
    ):
        """Initialize DDPG."""
        super().__init__(
            env_id=env_id, 
            cfgs=cfgs, 
            algo=algo, 
            wrapper_type=wrapper_type,
            )

    def compute_loss_v(self, data):
        r"""
        computing value loss

        Args:
            data (dict): data from replay buffer

        Returns:
            torch.Tensor
        """
        obs, act, rew, obs_next, done = (
            data['obs'],
            data['act'],
            data['rew'],
            data['obs_next'],
            data['done'],
        )
        q_value_list = self.actor_critic.critic(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ = self.ac_targ.actor.predict(obs, deterministic=False, need_log_prob=False)
            q_targ_list = self.ac_targ.critic(obs_next, act_targ)
            q_targ = torch.min(torch.vstack(q_targ_list), dim=0).values
            backup = rew + self.cfgs.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = sum([torch.mean((q_value - backup) ** 2) for q_value in q_value_list])
        q_value = sum(torch.mean(q_value) for q_value in q_value_list)
        # Useful info for logging
        q_info = dict(QVals=q_value.detach().numpy())
        return loss_q, q_info


