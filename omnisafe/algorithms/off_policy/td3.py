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
"""Implementation of the TD3 algorithm."""
from typing import Dict, NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.ddpg import DDPG


@registry.register
class TD3(DDPG):  # pylint: disable=too-many-instance-attributes
    """The Twin Delayed DDPG (TD3) algorithm.

    References:
        - Title: Addressing Function Approximation Error in Actor-Critic Methods
        - Authors: Scott Fujimoto, Herke van Hoof, David Meger.
        - URL: `TD3 <https://arxiv.org/abs/1802.09477>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize TD3."""
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )

    # pylint: disable-next=too-many-arguments
    def compute_loss_v(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        rew: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""Computing value loss.

        Detailedly, TD3 compute the loss by:

        .. math::
            L_1 = [Q^{C}_1 (s, a) - (r + \gamma (1-d) \underset{i = 1, 2}{\min} Q^{C}_i (s', \pi(s'))]^2\\
            L_2 = [Q^{C}_2 (s, a) - (r + \gamma (1-d) \underset{i = 1, 2}{\min} Q^{C}_i (s', \pi(s'))]^2

        .. note::

            TD3 uses two Q functions to reduce overestimation bias.
            In this function, we use the minimum of the two Q functions as the target Q value.

            Also, TD3 use action with noise to compute the target Q value.

        Args:
            obs (:class:`torch.Tensor`): ``observation`` saved in data.
            act (:class:`torch.Tensor`): ``action`` saved in data.
            rew (:class:`torch.Tensor`): ``reward`` saved in data.
            next_obs (:class:`torch.Tensor`): ``next observation`` saved in data.
            done (:class:`torch.Tensor`): ``terminated`` saved in data.
        """
        q_value_list = self.actor_critic.critic(obs, act)
        # Bellman backup for Q function
        with torch.no_grad():
            act_targ = self.ac_targ.actor.predict(obs, deterministic=True, need_log_prob=False)
            q_targ = torch.min(torch.vstack(self.ac_targ.critic(next_obs, act_targ)), dim=0).values
            backup = rew + self.cfgs.gamma * (1 - done) * q_targ
        # MSE loss against Bellman backup
        loss_q = []
        q_values = []
        for q_value in q_value_list:
            loss_q.append(torch.mean((q_value - backup) ** 2))
            q_values.append(torch.mean(q_value))

        # Useful info for logging
        q_info = dict(QVals=sum(q_values).detach().numpy())
        return sum(loss_q), q_info
