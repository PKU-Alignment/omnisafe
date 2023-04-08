# Copyright 2022-2023 OmniSafe Team. All Rights Reserved.
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
"""On-policy algorithms."""

from omnisafe.algorithms.on_policy import (
    base,
    first_order,
    naive_lagrange,
    penalty_function,
    saute,
    second_order,
    simmer,
)
from omnisafe.algorithms.on_policy.base import PPO, TRPO, NaturalPG, PolicyGradient

# from omnisafe.algorithms.on_policy.early_terminated import PPOEarlyTerminated, PPOLagEarlyTerminated
from omnisafe.algorithms.on_policy.first_order import CUP, FOCOPS
from omnisafe.algorithms.on_policy.naive_lagrange import PDO, RCPO, OnCRPO, PPOLag, TRPOLag
from omnisafe.algorithms.on_policy.penalty_function import IPO, P3O
from omnisafe.algorithms.on_policy.saute import TRPOSaute
from omnisafe.algorithms.on_policy.second_order import CPO, PCPO
from omnisafe.algorithms.on_policy.simmer import TRPOSimmerPID


# from omnisafe.algorithms.on_policy.pid_lagrange import CPPOPid, TRPOPid


# from omnisafe.algorithms.on_policy.simmer import (
#     PPOLagSimmerPid,
#     PPOLagSimmerQ,
#     PPOSimmerPid,
#     PPOSimmerQ,
# )


__all__ = [
    *base.__all__,
    # *early_terminated.__all__,
    *first_order.__all__,
    *naive_lagrange.__all__,
    *penalty_function.__all__,
    # *pid_lagrange.__all__,
    *saute.__all__,
    *second_order.__all__,
    *simmer.__all__,
]
