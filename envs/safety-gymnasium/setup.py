#!/usr/bin/env python3
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
"""Set up."""

import pathlib
import re
import sys

from setuptools import setup


HERE = pathlib.Path(__file__).absolute().parent
VERSION_FILE = HERE / 'safety_gymnasium' / 'version.py'

sys.path.insert(0, str(VERSION_FILE.parent))
import version  # noqa


VERSION_CONTENT = None

try:
    if not version.__release__:
        try:
            VERSION_CONTENT = VERSION_FILE.read_text(encoding='UTF-8')
            VERSION_FILE.write_text(
                data=re.sub(
                    r"""__version__\s*=\s*('[^']+'|"[^"]+")""",
                    r"__version__ = '{}'".format(version.__version__),
                    string=VERSION_CONTENT,
                ),
                encoding='UTF-8',
            )
        except OSError:
            VERSION_CONTENT = None

    setup(
        name='safety_gymnasium',
        version=version.__version__,
        install_requires=[
            'gymnasium==0.26.3',
            'pygame==2.1.0',
            'mujoco==2.3.0',
        ],
    )
finally:
    if VERSION_CONTENT is not None:
        with VERSION_FILE.open(mode='wt', encoding='UTF-8', newline='') as file:
            file.write(VERSION_CONTENT)
