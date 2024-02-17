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
"""Implementation of VAE."""

from typing import Dict, List, Optional

import torch
from torch.distributions import Distribution

from omnisafe.models.base import Actor
from omnisafe.models.diffuser import GaussianInvDynDiffusion, TemporalUnet
from omnisafe.typing import OmnisafeSpace


class DecisionDiffuserActor(Actor):
    """Class for DecisionDiffusor.

    BLAH BLAH BLAH

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list): List of hidden layer sizes.
        latent_dim (Optional[int]): Latent dimension, if None, latent_dim = act_dim * 2.
        activation (Activation): Activation function.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        horizon: int,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        cls_free_cond_dim: int,
    ) -> None:
        """Initialize an instance of :class:`VAE`."""
        super().__init__(
            obs_space,
            act_space,
            [],
        )
        self._horizon = horizon
        self._cls_free_cond_dim = cls_free_cond_dim
        temporal_unet = TemporalUnet(
            horizon=self._horizon,
            transition_dim=self._obs_dim,
            cls_free_condition_dim=cls_free_cond_dim,
            condition_dropout=0.25,
        )
        self._model = GaussianInvDynDiffusion(
            temporal_unet,
            horizon=self._horizon,
            observation_dim=self._obs_dim,
            action_dim=self._act_dim,
            cls_free_condition_guidance_w=1.2,
        )

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, obs: torch.Tensor) -> Distribution:
        """Forward is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """log_prob is not used in this method, it is just for compatibility."""
        raise NotImplementedError

    def predict(  # pylint: disable=unused-argument
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        cls_free_condition_list: Optional[List[torch.Tensor]] = None,
        extra_state_condition: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Predict the action given observation.

        deterministic if not used in VAE model. VAE actor's default behavior is stochastic,
        sampling from the latent standard normal distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            torch.Tensor: Predicted action.
        """
        if not cls_free_condition_list:
            cls_free_condition_list = [
                torch.zeros(self._cls_free_cond_dim),
            ]  # use reward as classifier free condition
        state_conditions = {0: obs}
        if extra_state_condition is not None:
            state_conditions.update(extra_state_condition)
        predicted_trajectory = self.model.conditional_sample(
            state_conditions,
            cls_free_condition_list=cls_free_condition_list,
        )
        obs_comb = torch.cat([predicted_trajectory[:, 0, :], predicted_trajectory[:, 1, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, 2 * self._obs_dim)
        action = self.model.inv_model(obs_comb)
        return action.flatten()
