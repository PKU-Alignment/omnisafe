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
"""Test Utils"""

import helpers
from omnisafe.utils.core import discount_cumsum_torch
from omnisafe.utils.distributed_utils import mpi_fork, mpi_statistics_scalar
from omnisafe.utils.logger_utils import convert_json, colorize
from omnisafe.utils.tools import to_ndarray
from omnisafe.common.experiment_grid import ExperimentGrid
import torch
import numpy as np


@helpers.parametrize(
    item=[1, 1.0, [1, 2, 3], (1, 2, 3), {'a': 1, 'b': 2}, torch.tensor([1, 2, 3])]
)
def test_to_ndarray(item):
    """Test to_ndarray"""
    if isinstance(item, torch.Tensor):
        assert isinstance(to_ndarray(item), np.ndarray)
    elif isinstance(item, list):
        out_list = to_ndarray(item)
        for val in out_list:
            assert isinstance(val, np.ndarray)
    elif isinstance(item, tuple):
        out_tuple = to_ndarray(item)
        for val in out_tuple:
            assert isinstance(val, np.ndarray)
    elif isinstance(item, dict):
        out_dict = to_ndarray(item)
        for val in out_dict.values():
            assert isinstance(val, np.ndarray)
    else:
        assert isinstance(to_ndarray(item), np.ndarray)

def get_answer(gamma: float) -> torch.Tensor:
    """Input gamma and return the answer"""
    if gamma == 0.9:
        return torch.tensor([11.4265, 11.5850, 10.6500,  8.5000,  5.0000], dtype=torch.float64)
    elif gamma == 0.99:
        return torch.tensor([14.6045, 13.7419, 11.8605,  8.9500,  5.0000], dtype=torch.float64)
    elif gamma == 0.999:
        return torch.tensor([14.9600, 13.9740, 11.9860,  8.9950,  5.0000], dtype=torch.float64)

@helpers.parametrize(
    discount=[0.9, 0.99, 0.999],
)
def test_discount_cumsum_torch(
    discount: float,
):
    """Test discount_cumsum_torch"""
    x1=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y1=get_answer(discount) 
    assert torch.allclose(discount_cumsum_torch(x1, discount), y1), 'discount_cumsum_torch is not correct'

def test_distributed_tools():
    """Test mpi_fork"""
    mpi_fork(2, test_message=['examples/train_from_custom_dict.py', '--parallel', '2'])

@helpers.parametrize(
    obj=[{'a': 1, 'b': 2}, [1, 2, 3], ('a', 'b', 'c')],
)
def test_convert_json(obj):
    """Test convert_json"""
    assert convert_json(obj) == obj

@helpers.parametrize(
    message=['hello'],
    color=['red', 'green'],
    bold=[True, False],
    highlight=[True, False],
)
def test_colorize(message, color, bold, highlight):
    """Test colorize"""
    colorize(message, color, bold, highlight)
    
