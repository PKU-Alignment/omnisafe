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
"""Test ensemble."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from omnisafe.algorithms.model_based.base.ensemble import (
    EnsembleFC,
    EnsembleModel,
    StandardScaler,
    unbatched_forward,
)


def test_standard_scaler():
    standard_scaler = StandardScaler(device='cpu')
    torch_input = torch.rand(10, 10)
    assert isinstance(standard_scaler.transform(torch_input), torch.Tensor)


def test_unbatched_forward():
    layer = nn.Linear(10, 10)
    torch_input = torch.rand(10, 10)
    assert isinstance(unbatched_forward(layer, torch_input, 0), torch.Tensor)


def test_ensemble_fc():
    ensemble_fc = EnsembleFC(
        in_features=10,
        out_features=10,
        ensemble_size=10,
        weight_decay=0.0,
        bias=False,
    )
    torch_input = torch.rand(10, 10, 10)
    assert isinstance(ensemble_fc(torch_input), torch.Tensor)


def test_enemble_model():
    ensemble_model = EnsembleModel(
        device='cpu',
        state_size=5,
        action_size=5,
        reward_size=1,
        cost_size=1,
        ensemble_size=10,
        predict_reward=True,
        predict_cost=True,
    )
    numpy_state = np.random.rand(10, 10, 10)
    mean, var = ensemble_model(numpy_state)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(var, torch.Tensor)
    mean, log_var = ensemble_model(numpy_state, ret_log_var=True)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(log_var, torch.Tensor)
    numpy_state = np.random.rand(1, 10, 10)
    mean, var = ensemble_model.forward_idx(numpy_state, idx_model=0)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(var, torch.Tensor)
    mean, log_var = ensemble_model.forward_idx(numpy_state, idx_model=0, ret_log_var=True)
    assert isinstance(mean, torch.Tensor)
    assert isinstance(log_var, torch.Tensor)
