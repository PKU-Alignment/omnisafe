# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""Some Core Functions"""

import torch


def discount_cumsum_torch(x_vector: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the discounted cumulative sum of vectors."""
    length = x_vector.shape[0]
    x_vector = x_vector.type(torch.float64)
    for idx in reversed(range(length)):
        if idx == length - 1:
            cumsum = x_vector[idx]
        else:
            cumsum = x_vector[idx] + discount * cumsum
        x_vector[idx] = cumsum
    return x_vector
