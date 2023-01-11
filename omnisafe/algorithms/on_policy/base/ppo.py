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
"""Implementation of the PPO algorithm."""

from typing import NamedTuple, Tuple

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient


@registry.register
class PPO(PolicyGradient):
    """The Proximal Policy Optimization (PPO) algorithm.

    References:
        - Title: Proximal Policy Optimization Algorithms
        - Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        - URL: `PPO <https://arxiv.org/abs/1707.06347>`_
    """

    def __init__(self, env_id: str, cfgs: NamedTuple) -> None:
        """Initialize Proximal Policy Optimization .

        .. note::
            The ``clip`` parameter is the clip parameter in PPO,
            which is used to clip the ratio of the new policy and the old policy.
            The ``clip`` parameter is set to 0.2 in the original paper.

        Args:
            env_id (str): The environment id.
            cfgs (NamedTuple): The configuration of the algorithm.
        """
        self.clip = cfgs.clip
        super().__init__(
            env_id=env_id,
            cfgs=cfgs,
        )

    # pylint: disable-next=too-many-arguments
    def compute_loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        log_p: torch.Tensor,
        adv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Computing pi/actor loss.
        In Proximal Policy Optimization, the loss is defined as:

        .. math::
            L^{CLIP} = \mathbb{E}_{s_t \sim \rho_{\theta}}
            \left[ \min(r_t A^{R}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A^{R}_t) \right]

        where :math:`r_t = \frac{\pi_\theta(a_t|s_t)}{\pi_\theta^{old}(a_t|s_t)}`,
        :math:`\epsilon` is the clip parameter, :math:`A^{R}_t` is the advantage.

        Args:
            obs (torch.Tensor): ``observation`` stored in buffer.
            act (torch.Tensor): ``action`` stored in buffer.
            log_p (torch.Tensor): ``log probability`` of action stored in buffer.
            adv (torch.Tensor): ``advantage`` stored in buffer.
            cost_adv (torch.Tensor): ``cost advantage`` stored in buffer.
        """
        dist, _log_p = self.actor_critic.actor(obs, act)
        # importance ratio
        ratio = torch.exp(_log_p - log_p)
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
        loss_pi = -(torch.min(ratio * adv, ratio_clip * adv))
        loss_pi += self.cfgs.entropy_coef * dist.entropy().mean()
        # useful extra info
        approx_kl = (0.5 * (dist.mean - act) ** 2 / dist.stddev**2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

        return loss_pi.mean(), pi_info
