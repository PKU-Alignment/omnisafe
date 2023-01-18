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
"""tools for experiment grid"""

import os

import numpy as np
import torch
import yaml
import string


def all_bools(vals):
    return all([isinstance(v, bool) for v in vals])


def valid_str(v):
    """
    Convert a value or values to a string which could go in a filepath.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    """
    if hasattr(v, '__name__'):
        return valid_str(v.__name__)

    if isinstance(v, tuple) or isinstance(v, list):
        return '-'.join([valid_str(x) for x in v])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(v).lower()
    valid_chars = "-_%s%s" % (string.ascii_letters, string.digits)
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v
