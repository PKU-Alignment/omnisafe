import numpy as np
import torch

from omnisafe.models.actor_q_critic import ActorQCritic
from omnisafe.models.critic.q_critic import QCritic


# observation_space,
# action_space,
# pi_type,
# standardized_obs: bool,
# shared_weights: bool,
# ac_kwargs: dict,


class ConstraintActorQCritic(ActorQCritic):
    def __init__(
        self,
        observation_space,
        action_space,
        standardized_obs: bool,
        scale_rewards: bool,
        model_cfgs,
    ):
        #         observation_space,
        # action_space,
        # standardized_obs: bool,
        # shared_weights: bool,
        # model_cfgs,
        # weight_initialization_mode='kaiming_uniform',
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            standardized_obs=standardized_obs,
            shared_weights=model_cfgs.shared_weights,
            model_cfgs=model_cfgs,
        )
        self.cost_critic = QCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim,
            hidden_sizes=self.ac_kwargs.val.hidden_sizes,
            activation=self.ac_kwargs.val.activation,
            weight_initialization_mode=model_cfgs.weight_initialization_mode,
            shared=model_cfgs.shared_weights,
        )

    def step(self, obs, deterministic=False):
        """
        If training, this includes exploration noise!
        Expects that obs is not pre-processed.
        Args:
            obs, description
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
            a, logp_a = self.actor.predict(obs, deterministic=deterministic)
            v = self.critic(obs, a)
            c = self.cost_critic(obs, a)
            a = np.clip(a.numpy(), -self.act_limit, self.act_limit)

        return a, v.numpy(), c.numpy(), logp_a.numpy()
