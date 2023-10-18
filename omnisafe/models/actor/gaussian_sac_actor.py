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
"""Implementation of GaussianStdNetActor."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from omnisafe.models.base import Actor
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.math import TanhNormal
from omnisafe.utils.model import build_mlp_network


class GaussianSACActor(Actor):
    """Implementation of GaussianSACActor.

    GaussianSACActor is a Gaussian actor with a learnable standard deviation network.
    It is used in ``SAC``, and other offline or model-based algorithms related to ``SAC``.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    _log2: torch.Tensor
    _current_dist: Normal

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
    ) -> None:
        """Initialize an instance of :class:`GaussianSACActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)

        self.net: nn.Module = build_mlp_network(
            sizes=[self._obs_dim, *self._hidden_sizes, self._act_dim * 2],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

        self._current_raw_action: torch.Tensor | None = None
        self.register_buffer('_log2', torch.log(torch.tensor(2.0)))

    def _distribution(self, obs: torch.Tensor) -> Normal:
        """Get the distribution of the actor.

        .. warning::
            This method is not supposed to be called by users. You should call :meth:`forward`
            instead.

        **Specifically, this method will clip the standard deviation to a range of [-20, 2].**

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The normal distribution of the mean and standard deviation from the actor.
        """
        mean, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = log_std.exp()
        return Normal(mean, std)

    def predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Predict the action given observation.

        The predicted action depends on the ``deterministic`` flag.

        - If ``deterministic`` is ``True``, the predicted action is the mean of the distribution.
        - If ``deterministic`` is ``False``, the predicted action is sampled from the distribution.

        Args:
            obs (torch.Tensor): Observation from environments.
            deterministic (bool, optional): Whether to use deterministic policy. Defaults to False.

        Returns:
            The mean of the distribution if deterministic is True, otherwise the sampled action.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True

        action = self._current_dist.mean if deterministic else self._current_dist.rsample()

        self._current_raw_action = action

        return torch.tanh(action)

    def forward(self, obs: torch.Tensor) -> TanhNormal:
        """Forward method.

        Args:
            obs (torch.Tensor): Observation from environments.

        Returns:
            The current distribution.
        """
        self._current_dist = self._distribution(obs)
        self._after_inference = True
        return TanhNormal(self._current_dist.mean, self._current_dist.stddev)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        # pylint: disable=not-callable
        r"""Compute the log probability of the action given the current distribution.

        .. warning::
            You must call :meth:`forward` or :meth:`predict` before calling this method.

        .. note::
            In this method, we will regularize the log probability of the action. The regularization
            is as follows:

            .. math::

                \log prob = \log \pi (a|s) - \sum_{i=1}^n (2 \log 2 - a_i - \log (1 + e^{-2 a_i}))

            where :math:`a` is the action, :math:`s` is the observation, and :math:`n` is the
            dimension of the action.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`.

        Returns:
            Log probability of the action.
        """
        assert self._after_inference, 'log_prob() should be called after predict() or forward()'
        self._after_inference = False

        if self._current_raw_action is not None:
            logp = self._current_dist.log_prob(self._current_raw_action).sum(axis=-1)
            logp -= (
                2
                * (
                    self._log2
                    - self._current_raw_action
                    - nn.functional.softplus(
                        -2 * self._current_raw_action,
                    )
                )
            ).sum(
                axis=-1,
            )  # type: ignore
            self._current_raw_action = None
        else:
            logp = (
                TanhNormal(self._current_dist.mean, self._current_dist.stddev)
                .log_prob(act)
                .sum(axis=-1)
            )

        return logp

    @property
    def std(self) -> float:
        """Standard deviation of the distribution."""
        return self._current_dist.stddev.mean().item()

    @std.setter
    def std(self, std: float) -> None:
        raise NotImplementedError('GaussianStdNetActor does not support setting std.')
