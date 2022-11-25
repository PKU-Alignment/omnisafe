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

import torch

from omnisafe.algos.models.actor_critic import ActorCritic
from omnisafe.algos.models.critic import Critic


class ConstraintActorCritic(ActorCritic):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.c = Critic(obs_dim=self.obs_shape[0], shared=None, **self.ac_kwargs['val'])

    def step(self, obs, deterministic=False):
        """Produce action, value, log_prob(action).
        If training, this includes exploration noise!

        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: do the updates at the end of batch!
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.v(obs)
            c = self.c(obs)

            a, logp_a = self.pi.predict(obs, deterministic=deterministic)

        return a.numpy(), v.numpy(), c.numpy(), logp_a.numpy()
