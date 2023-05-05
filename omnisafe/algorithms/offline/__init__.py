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
"""Offline algorithms."""

from omnisafe.algorithms.offline.bcq import BCQ
from omnisafe.algorithms.offline.bcq_lag import BCQLag
from omnisafe.algorithms.offline.c_crr import CCRR
from omnisafe.algorithms.offline.coptidice import COptiDICE
from omnisafe.algorithms.offline.crr import CRR
from omnisafe.algorithms.offline.vae_bc import VAEBC


__all__ = ['BCQ', 'BCQLag', 'CCRR', 'CRR', 'COptiDICE', 'VAEBC']
