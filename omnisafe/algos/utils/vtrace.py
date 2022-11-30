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
"""vtrace"""

import numpy as np


def calculate_v_trace(
    policy_action_probs: np.ndarray,
    values: np.ndarray,  # including bootstrap
    rewards: np.ndarray,  # including bootstrap
    behavior_action_probs: np.ndarray,
    gamma=0.99,
    rho_bar=1.0,
    c_bar=1.0,
) -> tuple:
    """
    calculate V-trace targets for off-policy actor-critic learning recursively
    as proposed in: Espeholt et al. 2018, IMPALA

    :param policy_action_probs:
    :param values:
    :param rewards:
    :param bootstrap_value:
    :param behavior_action_probs:
    :param gamma:
    :param rho_bar:
    :param c_bar:
    :return: V-trace targets, shape=(batch_size, sequence_length)
    """
    assert values.ndim == 1, 'Please provide 1d-arrays'
    assert rewards.ndim == 1
    assert policy_action_probs.ndim == 1
    assert behavior_action_probs.ndim == 1
    assert c_bar <= rho_bar

    sequence_length = policy_action_probs.shape[0]
    # print('sequence_length:', sequence_length)
    rhos = np.divide(policy_action_probs, behavior_action_probs)
    clip_rhos = np.minimum(rhos, rho_bar)
    clip_cs = np.minimum(rhos, c_bar)
    # values_plus_1 = np.append(values, bootstrap_value)

    v_s = np.copy(values[:-1])  # copy all values except bootstrap value
    # v_s = np.zeros_like(values)
    last_v_s = values[-1]  # bootstrap from last state

    # calculate v_s
    for index in reversed(range(sequence_length)):
        delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
        v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
        last_v_s = v_s[index]  # accumulate current v_s for next iteration

    # calculate q_targets
    v_s_plus_1 = np.append(v_s[1:], values[-1])
    policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

    return v_s, policy_advantage, clip_rhos
