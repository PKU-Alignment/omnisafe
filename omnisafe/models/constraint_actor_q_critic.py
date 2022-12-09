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
"""Implementation of ConstraintQActorCritic."""

import numpy as np
import torch

from omnisafe.models.actor_q_critic import Actor_Q_Critic
from omnisafe.models.critic.q_critic import QCritic


class ConstraintQActorCritic(Actor_Q_Critic):
    def __init__(self, **cfgs):
        super().__init__(**cfgs)
        self.c = QCritic(
            obs_dim=self.obs_shape[0], act_dim=self.act_dim, shared=None, **self.ac_kwargs['val']
        )

    def step(self, obs, determinstic=False):
        """
        If training, this includes exploration noise!
        Expects that obs is not pre-processed.
        Args:
            obs, , description
        Returns:
            action, value, log_prob(action)
        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            a, logp_a = self.pi.predict(obs, determinstic=determinstic)
            v = self.v(obs, a)
            c = self.c(obs, a)
            a = np.clip(a.numpy(), -self.act_limit, self.act_limit)

        return a, v.numpy(), c.numpy(), logp_a.numpy()
