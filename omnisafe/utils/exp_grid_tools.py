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

import string


def all_bools(vals):
    """check if all values are bools"""
    return all(isinstance(v, bool) for v in vals)


def valid_str(vals):
    """
    Convert a value or values to a string which could go in a path of file.

    Partly based on `this gist`_.

    .. _`this gist`: https://gist.github.com/seanh/93666

    """
    if hasattr(vals, '__name__'):
        return valid_str(vals.__name__)

    if isinstance(vals, (list, tuple)):
        return '-'.join([valid_str(x) for x in vals])

    # Valid characters are '-', '_', and alphanumeric. Replace invalid chars
    # with '-'.
    str_v = str(vals).lower()
    valid_chars = f'-_{string.ascii_letters}{string.digits}'
    str_v = ''.join(c if c in valid_chars else '-' for c in str_v)
    return str_v
