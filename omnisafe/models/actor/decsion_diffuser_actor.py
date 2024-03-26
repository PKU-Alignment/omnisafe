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
"""Implementation of DecisionDiffuserActor."""

from __future__ import annotations

import torch
from torch.distributions import Distribution

from omnisafe.models.base import Actor
from omnisafe.models.dd_models.diffusion import GaussianInvDynDiffusion
from omnisafe.models.dd_models.temporal import TemporalUnet
from omnisafe.typing import Activation, InitFunction, OmnisafeSpace
from omnisafe.utils.config import Config
from omnisafe.utils.model import initialize_layer


# pylint: disable-next=too-many-instance-attributes
class DecisionDiffuserActor(Actor):
    """Implementation of MLPActor.

    MLPActor is a Gaussian actor with a learnable mean value. It is used in off-policy algorithms
    such as ``DDPG``, ``TD3`` and so on.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        output_activation (Activation, optional): Output activation function. Defaults to ``'tanh'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
        self,
        obs_space: OmnisafeSpace,
        act_space: OmnisafeSpace,
        hidden_sizes: list[int],
        activation: Activation = 'relu',
        output_activation: Activation = 'tanh',
        weight_initialization_mode: InitFunction = 'kaiming_uniform',
        cfgs: Config = None,
    ) -> None:
        """Initialize an instance of :class:`MLPActor`."""
        super().__init__(obs_space, act_space, hidden_sizes, activation, weight_initialization_mode)
        self._cfgs = cfgs
        TUmodel = TemporalUnet(
            horizon=self._cfgs.algo_cfgs.horizon,
            transition_dim=self._obs_dim,
            dim_mults=self._cfgs.model_cfgs.temporalU_model.dim_mults,
            returns_condition=self._cfgs.model_cfgs.returns_condition,
            dim=self._cfgs.model_cfgs.temporalU_model.dim,
            condition_dropout=self._cfgs.model_cfgs.temporalU_model.condition_dropout,
            calc_energy=self._cfgs.model_cfgs.temporalU_model.calc_energy,
            constraints_dim=self._cfgs.algo_cfgs.constraints_dim,
            skills_dim=self._cfgs.algo_cfgs.skills_dim,
        )
        GDDModel = GaussianInvDynDiffusion(
            TUmodel,
            horizon=self._cfgs.algo_cfgs.horizon,
            observation_dim=self._obs_dim,
            action_dim=self._act_dim,
            n_timesteps=self._cfgs.algo_cfgs.n_diffusion_steps,
            clip_denoised=self._cfgs.model_cfgs.diffuser_model.clip_denoised,
            predict_epsilon=self._cfgs.model_cfgs.diffuser_model.predict_epsilon,
            hidden_dim=self._cfgs.model_cfgs.diffuser_model.hidden_dim,
            loss_discount=self._cfgs.model_cfgs.diffuser_model.loss_discount,
            returns_condition=self._cfgs.model_cfgs.returns_condition,
            train_only_inv=self._cfgs.model_cfgs.diffuser_model.train_only_inv,
            condition_guidance_w=self._cfgs.model_cfgs.diffuser_model.condition_guidance_w,
            history_length=self._cfgs.evaluate_cfgs.obs_history_length,
            multi_step_pred=self._cfgs.evaluate_cfgs.multi_step_pred,
        )
        for _name, layer in GDDModel.named_modules():
            if isinstance(layer, torch.nn.Linear):
                initialize_layer(self._cfgs.model_cfgs.weight_initialization_mode, layer)
        self.net = GDDModel
        self.multi_step_pred = self._cfgs.evaluate_cfgs.multi_step_pred
        self.obs_history_length = self._cfgs.evaluate_cfgs.obs_history_length
        self.init_step_config()

    def loss(
        self,
        x: torch.Tensor,
        returns: torch.Tensor = None,
        constraints: torch.Tensor = None,
        skills: torch.Tensor = None,
    ) -> tuple:
        """The progress of loss training."""
        return self.net.loss(x, returns, constraints, skills)

    def init_step_config(self) -> None:
        """Initialize config when each episode begin."""
        self.obs_history = []
        self.samples_history_length = 0
        self.step_count = 0
        self._eval_returns = self._cfgs.evaluate_cfgs.returns * torch.ones(
            1,
            1,
            dtype=torch.float32,
        )
        self._eval_constraints = torch.tensor(
            [self._cfgs.evaluate_cfgs.constraints],
            dtype=torch.float32,
        )
        self._eval_skills = torch.tensor([self._cfgs.evaluate_cfgs.skills], dtype=torch.float32)

    def update_step_config(self, obs: torch.Tensor) -> tuple:
        """Update config at each step."""
        use_sample = False
        self.obs_history.append(obs)
        if len(self.obs_history) > self.obs_history_length:
            self.obs_history.pop(0)
        if self.step_count % self.multi_step_pred == 0:
            use_sample = True
        obs = torch.stack(self.obs_history, dim=0)
        return obs, use_sample

    def predict(
        self,
        obs: torch.Tensor,
        deterministic: bool = True,
    ) -> torch.Tensor:
        """Predict action from observation.

        deterministic is not used in this method, it is just for compatibility.

        Args:
            obs (torch.Tensor): Observation.
            deterministic (bool, optional): Whether to return deterministic action. Defaults to False.

        Returns:
            torch.Tensor: Action.
        """
        if self.step_count % self._cfgs.algo_cfgs.steps_per_epoch == 0:
            self.init_step_config()
        obs, use_sample = self.update_step_config(obs)
        device = self.net.inv_model.state_dict()['0.weight'].device
        obs = obs.unsqueeze(0).to(device)
        returns = self._eval_returns.to(device)
        constraints = self._eval_constraints.to(device)
        skills = self._eval_skills.to(device)
        if use_sample:
            samples = self.net.conditional_sample(
                obs,
                returns=returns,
                constraints=constraints,
                skills=skills,
            )
            self.samples = samples.clone()
            self.sample_hisory_len = obs.shape[0]
        next_obs = self.samples[
            :,
            self.sample_hisory_len + self.step_count % self.multi_step_pred + 1,
            :,
        ]
        obs_comb = torch.cat([obs[:, -1, :], next_obs], dim=-1).reshape(1, -1)
        action = self.net.inv_model(obs_comb)
        self.step_count += 1
        return action

    def _distribution(self, obs: torch.Tensor) -> Distribution:
        raise NotImplementedError

    def forward(self, *args: tuple, **kwargs: dict) -> torch.Tensor:
        """Forward function of decision diffuser actor."""
        return self.net.conditional_sample(*args, **kwargs)

    def log_prob(self, act: torch.Tensor) -> torch.Tensor:
        """Log probability of the action.

        Args:
            act (torch.Tensor): Action from :meth:`predict` or :meth:`forward`  tensor.

        Raises:
            NotImplementedError: The method is not implemented.
        """
        raise NotImplementedError
