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
"""Lagrange"""

import abc
import torch
import torch.optim as optim
from torch.nn.functional import softplus


class Lagrange(abc.ABC):
    """
    Abstract base class for Lagrangian-base Algorithm
    """

    def __init__(
        self,
        cost_limit: float,
        lagrangian_multiplier_init: float,
        lambda_lr: float,
        lambda_optimizer: str,
        beta: float,
        device: str,
    ):
        self.cost_limit = cost_limit
        self.lambda_lr = lambda_lr
        self.beta = beta
        init_value = max(lagrangian_multiplier_init, 1e-5)
        # fetch optimizer from PyTorch optimizer package
        assert hasattr(optim, lambda_optimizer), f'Optimizer={lambda_optimizer} not found in torch.'
        torch_opt = getattr(optim, lambda_optimizer)
        self.lagrangian_multiplier = torch.tensor(init_value, device=device,requires_grad=True).float()
        self.penalty_optimizer = torch_opt(
            [
                self.lagrangian_multiplier,
            ],
            lr=self.lambda_lr ,
        )
        
    def compute_lambda_loss(self, ep_costs):
        """Penalty loss for Lagrange multiplier."""
        return -self.lagrangian_multiplier * (ep_costs - self.cost_limit*self.beta)

    def update_lagrange_multiplier(self, ep_costs):
        """Update Lagrange multiplier (lambda)
        Note: ep_costs obtained from: self.logger.get_stats('EpCosts')[0]
        are already averaged across MPI processes.
        """
        self.penalty_optimizer.zero_grad()
        lambda_loss = self.compute_lambda_loss(ep_costs)
        lambda_loss.backward()
        self.penalty_optimizer.step()
        #self.lagrangian_multiplier.data.clamp_(0)  # enforce: lambda in [0, inf]
