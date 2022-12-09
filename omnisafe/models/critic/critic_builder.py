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
"""Implementation of CriticBuilder."""

import torch.nn as nn

from omnisafe.models.critic.q_critic import QCritic
from omnisafe.models.critic.v_critic import VCritic
from omnisafe.utils.model_utils import Activation, InitFunction


# pylint: disable=too-few-public-methods
class CriticBuilder:
    """Implementation of CriticBuilder."""

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: list,
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        shared: nn.Module = None,
    ) -> None:
        """Initialize CriticBuilder."""
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.weight_initialization_mode = weight_initialization_mode
        self.shared = shared

    def build_critic(self, critic_type: str):
        """Build critic."""
        if critic_type == 'q':
            return QCritic(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
            )
        if critic_type == 'v':
            return VCritic(
                obs_dim=self.obs_dim,
                act_dim=self.act_dim,
                hidden_sizes=self.hidden_sizes,
                activation=self.activation,
                weight_initialization_mode=self.weight_initialization_mode,
                shared=self.shared,
            )

        raise NotImplementedError(f'critic_type "{critic_type}" is not implemented.')
