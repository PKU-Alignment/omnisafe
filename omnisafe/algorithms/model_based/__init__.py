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
"""Model-Based algorithms."""

from omnisafe.algorithms.model_based import base
from omnisafe.algorithms.model_based.base import LOOP, PETS
from omnisafe.algorithms.model_based.cap_pets import CAPPETS
from omnisafe.algorithms.model_based.cce_pets import CCEPETS
from omnisafe.algorithms.model_based.rce_pets import RCEPETS
from omnisafe.algorithms.model_based.safeloop import SafeLOOP


__all__ = [
    *base.__all__,
    'CAPPETS',
    'CCEPETS',
    'SafeLOOP',
    'RCEPETS',
]
