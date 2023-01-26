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

from typing import Tuple

import torch


# pylint: disable-next=too-many-arguments, too-many-locals
def calculate_v_trace(
    policy_action_probs: torch.Tensor,
    values: torch.Tensor,  # including bootstrap
    rewards: torch.Tensor,  # including bootstrap
    behavior_action_probs: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,]:
    r"""This function is used to calculate V-trace targets.

    .. math::
        A_t = \sum_{k=0}^{n-1} (\lambda \gamma)^k \delta_{t+k} +
        (\lambda \gamma)^n * \rho_{t+n} * (1 - d_{t+n}) * (V(x_{t+n}) - b_{t+n})

    Calculate V-trace targets for off-policy actor-critic learning recursively.
    For more details,
    please refer to the paper: `Espeholt et al. 2018, IMPALA <https://arxiv.org/abs/1802.01561>`_.

    Args:
        policy_action_probs (torch.Tensor): action probabilities of policy network, shape=(sequence_length,)
        values (torch.Tensor): state values, shape=(sequence_length+1,)
        rewards (torch.Tensor): rewards, shape=(sequence_length+1,)
        behavior_action_probs (torch.Tensor): action probabilities of behavior network, shape=(sequence_length,)
        gamma (float): discount factor
        rho_bar (float): clip rho
        c_bar (float): clip c

    Returns:
        tuple: V-trace targets, shape=(batch_size, sequence_length)
    """
    assert values.ndim == 1, 'Please provide 1d-arrays'
    assert rewards.ndim == 1
    assert policy_action_probs.ndim == 1
    assert behavior_action_probs.ndim == 1
    assert c_bar <= rho_bar

    sequence_length = policy_action_probs.shape[0]
    # pylint: disable-next=assignment-from-no-return
    rhos = torch.div(policy_action_probs, behavior_action_probs)
    clip_rhos = torch.min(
        rhos, torch.as_tensor(rho_bar)
    )  # pylint: disable=assignment-from-no-return
    clip_cs = torch.min(rhos, torch.as_tensor(c_bar))  # pylint: disable=assignment-from-no-return
    v_s = values[:-1].clone()  # copy all values except bootstrap value
    last_v_s = values[-1]  # bootstrap from last state

    # calculate v_s
    for index in reversed(range(sequence_length)):
        delta = clip_rhos[index] * (rewards[index] + gamma * values[index + 1] - values[index])
        v_s[index] += delta + gamma * clip_cs[index] * (last_v_s - values[index + 1])
        last_v_s = v_s[index]  # accumulate current v_s for next iteration

    # calculate q_targets
    v_s_plus_1 = torch.cat((v_s[1:], values[-1:]))
    policy_advantage = clip_rhos * (rewards[:-1] + gamma * v_s_plus_1 - values[:-1])

    return v_s, policy_advantage, clip_rhos
