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
"""Test normalizer."""

import torch

import helpers
from omnisafe.common.normalizer import Normalizer


@helpers.parametrize(
    shape=[(), (10,), (10, 10)],
)
def test_normalizer(shape: tuple):
    norm = Normalizer(shape)

    assert norm.mean.shape == shape

    data_lst = []
    for _ in range(1000):
        data = torch.randn(shape)
        data_lst.append(data)
        norm(data)

    data = torch.stack(data_lst)
    assert torch.allclose(data.mean(dim=0), norm.mean, atol=1e-2)
    assert torch.allclose(data.std(dim=0), norm.std, atol=1e-2)

    norm = Normalizer(shape)

    data_lst = []
    for _ in range(1000):
        data = torch.randn(10, *shape)
        data_lst.append(data)
        norm(data)

    data = torch.cat(data_lst)
    assert torch.allclose(data.mean(dim=0), norm.mean, atol=1e-2)
    assert torch.allclose(data.std(dim=0), norm.std, atol=1e-2)
