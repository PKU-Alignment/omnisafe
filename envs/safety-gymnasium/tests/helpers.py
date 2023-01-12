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
"""A parameterize decorator that allows for multiple dtypes to be tested."""

import itertools

import pytest  # pylint: disable=import-error


def parametrize(**argvalues) -> pytest.mark.parametrize:
    """A parameterize decorator that allows for multiple dtypes to be tested."""
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
