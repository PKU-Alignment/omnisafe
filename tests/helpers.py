import copy
import itertools
import os
import random
from typing import Iterable, Optional, Tuple, Union

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.types
from torch.utils import data


def dtype_numpy2torch(dtype: np.dtype) -> torch.dtype:
    return torch.tensor(np.zeros(1, dtype=dtype)).dtype


def dtype_torch2numpy(dtype: torch.dtype) -> np.dtype:
    return torch.zeros(1, dtype=dtype).numpy().dtype


def parametrize(**argvalues) -> pytest.mark.parametrize:
    arguments = list(argvalues)

    if 'dtype' in argvalues:
        dtypes = argvalues['dtype']
        argvalues['dtype'] = dtypes[:1]
        arguments.remove('dtype')
        arguments.insert(0, 'dtype')

        argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))
        first_product = argvalues[0]
        argvalues.extend((dtype,) + first_product[1:] for dtype in dtypes[1:])
    else:
        argvalues = list(itertools.product(*tuple(map(argvalues.get, arguments))))

    ids = tuple(
        '-'.join(f'{arg}({val})' for arg, val in zip(arguments, values)) for values in argvalues
    )

    return pytest.mark.parametrize(arguments, argvalues, ids=ids)
