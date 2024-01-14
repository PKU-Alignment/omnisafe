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
"""Adapter for the environment and the algorithm."""

from omnisafe.adapter.early_terminated_adapter import EarlyTerminatedAdapter
from omnisafe.adapter.modelbased_adapter import ModelBasedAdapter
from omnisafe.adapter.offline_adapter import OfflineAdapter
from omnisafe.adapter.offpolicy_adapter import OffPolicyAdapter
from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.adapter.onpolicy_adapter import OnPolicyAdapter
from omnisafe.adapter.saute_adapter import SauteAdapter
from omnisafe.adapter.simmer_adapter import SimmerAdapter
