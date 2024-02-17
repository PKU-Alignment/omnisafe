# Copyright 2022-2024 OmniSafe Team. All Rights Reserved.
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
"""A class representing the Exponential Moving Average (EMA).

EMA is a statistical calculation used to analyze data points by creating a series of averages of different
subsets of the full data set. It is commonly used in machine learning to update model parameters with a
weighted average of the current parameters and the previous average.
"""

import torch
from torch import nn


class EMA:
    """A class representing the Exponential Moving Average (EMA).

    EMA is a statistical calculation used to analyze data points by creating a series of averages of different
    subsets of the full data set. It is commonly used in machine learning to update model parameters with a
    weighted average of the current parameters and the previous average.

    Args:
        beta (float): The smoothing factor for the EMA calculation.

    Attributes:
        beta (float): The smoothing factor for the EMA calculation.

    Methods:
        update_model_average(ma_model, current_model): Update the model average parameters using exponential
        moving average.
        update_average(old, new): Update the average value using exponential moving average.
    """

    def __init__(self, beta: float) -> None:
        """Initialize the EMA object.

        Args:
            beta (float): The smoothing factor for the EMA calculation.
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model: nn.Module, current_model: nn.Module) -> None:
        """Update the model average parameters using exponential moving average.

        Args:
            ma_model (nn.Module): The model with the moving average parameters.
            current_model (nn.Module): The model with the current parameters.

        Returns:
            None
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Updates the average value using exponential moving average.

        Args:
            old (torch.Tensor): The previous average value.
            new (torch.Tensor): The new value to be incorporated into the average.

        Returns:
            torch.Tensor: The updated average value.
        """
        return old * self.beta + (1 - self.beta) * new
