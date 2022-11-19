import numpy as np
import torch

from omnisafe.algos.models.actor_q_critic import Actor_Q_Critic
from omnisafe.algos.models.q_critic import Q_Critic


class ConstraintQActorCritic(Actor_Q_Critic):
    def __init__(self, **cfgs):
        super().__init__(**cfgs)
        self.c = Q_Critic(
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
