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
"""Implementation of model in DICE algo family."""

from __future__ import annotations

import torch
from gymnasium import spaces
from torch import nn

from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.model import build_mlp_network


class ObsEncoder(nn.Module):
    """Implementation of observation encoder.

    Observation encoder is used to encode observation into a latent vector.
    It is similar to the QCritic, but the output dimension is not limited to 1.
    DICE-based algorithms often use the network like this to encode observation.

    Args:
        obs_space (OmnisafeSpace): observation space.
        act_space (OmnisafeSpace): action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
        out_dim (int, optional): Output dimension. Defaults to 1.
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        out_dim: int = 1,
    ) -> None:
        """Initialize an instance of :class:`ObsEncoder`."""
        nn.Module.__init__(self)

        if isinstance(obs_space, spaces.Box) and len(obs_space.shape) == 1:
            self._obs_dim = obs_space.shape[0]
        else:
            raise NotImplementedError

        if isinstance(act_space, spaces.Box) and len(act_space.shape) == 1:
            self._act_dim = act_space.shape[0]
        else:
            raise NotImplementedError

        self._out_dim = out_dim
        self.weight_initialization_mode = weight_initialization_mode
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        self.net = build_mlp_network(
            [self._obs_dim, *list(hidden_sizes), self._out_dim],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        When ``out_dim`` is 1, the output is squeezed to remove the last dimension.

        Args:
            obs (torch.Tensor): Observation.
        """
        if self._out_dim == 1:
            return self.net(obs).squeeze(-1)
        return self.net(obs)
