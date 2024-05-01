# Copyright 2024 OmniSafe Team. All Rights Reserved.
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
"""Models for CRABS."""
# pylint: disable=all
import abc
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.utils.math import TanhNormal


class EnsembleModel(pl.LightningModule):
    """Ensemble model for transition dynamics.

    Args:
        models (List[nn.Module]): List of transition models.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        """Initialize the ensemble model."""
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.n_elites = self.n_models
        self.elites = []  # type: ignore
        self.recompute_elites()
        self.automatic_optimization = False

    def recompute_elites(self):
        """Recompute the elites."""
        self.elites = list(range(len(self.models)))

    def forward(self, states, actions):
        """Forward pass of the ensemble model.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.

        Returns:
            torch.Tensor: The next states.
        """
        n = len(states)

        perm = np.random.permutation(n)
        inv_perm = np.argsort(perm)

        next_states = []
        for _i, (model_idx, indices) in enumerate(
            zip(self.elites, np.array_split(perm, len(self.elites))),
        ):
            next_states.append(self.models[model_idx](states[indices], actions[indices]))
        return torch.cat(next_states, dim=0)[inv_perm]

    def get_nlls(self, states, actions, next_states):
        """Get the negative log likelihoods.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.
            next_states (torch.Tensor): The next states.

        Returns:
            List[float]: The negative log likelihoods.
        """
        ret = []
        for model in self.models:
            distribution = model(states, actions, det=False)
            nll = -distribution.log_prob(next_states).mean().item()
            ret.append(nll)
        return ret

    def training_step(self, batch):
        """Training step of the ensemble model.

        Args:
            batch (dict[str, torch.Tensor]): The batch data.
        """
        total_loss = 0
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch, gp=False)
            total_loss = total_loss + loss
            self.log(f'model/{i}/training_loss', loss.item())

        opt = self.optimizers()
        opt.zero_grad()  # type: ignore

        self.manual_backward(total_loss)  # type: ignore
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()  # type: ignore

    def validation_step(self, batch):
        """Validation step of the ensemble model.

        Args:
            batch (dict[str, torch.Tensor]): The batch data.
        """
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch)
            self.log(f'model/{i}/val_loss', loss.item(), on_step=False)

    def configure_optimizers(self):
        """Configure the optimizers for the ensemble model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)


class MultiLayerPerceptron(nn.Sequential):
    """Multi-layer perceptron.

    Args:
        n_units (List[int]): The number of units in each layer.
        activation (nn.Module, optional): The activation function. Defaults to nn.ReLU.
        auto_squeeze (bool, optional): Whether to auto-squeeze the output. Defaults to True.
        output_activation ([type], optional): The output activation function. Defaults to None.
    """

    def __init__(
        self,
        n_units,
        activation=nn.ReLU,
        auto_squeeze=True,
        output_activation=None,
    ) -> None:
        """Initialize the multi-layer perceptron."""
        layers = []  # type: ignore
        for in_features, out_features in zip(n_units[:-1], n_units[1:]):
            if layers:
                layers.append(activation())
            layers.append(nn.Linear(in_features, out_features))
        if output_activation:
            layers.append(output_activation())
        super().__init__(*layers)

        self._n_units = n_units
        self._auto_squeeze = auto_squeeze
        self._activation = [activation]  # to prevent nn.Module put it into self._modules

    def forward(self, *inputs):
        """Forward pass of the MLP.

        Args:
            *inputs: The input tensors.
        """
        inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

        outputs = inputs
        for layer in self:
            outputs = layer(outputs)

        if self._auto_squeeze and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def copy(self):
        """Copy the MLP.

        Returns:
            MultiLayerPerceptron: The copied MLP.
        """
        return MultiLayerPerceptron(self._n_units, self._activation[0], self._auto_squeeze)

    def extra_repr(self):
        """Extra representation of the MLP.

        Returns:
            str: The extra representation.
        """
        return f'activation = {self._activation}, # units = {self._n_units}, squeeze = {self._auto_squeeze}'


class TransitionModel(pl.LightningModule):
    """Transition model for dynamics.

    Args:
        dim_state (int): The dimension of the state.
        normalizer (Normalizer): The observation normalizer.
        n_units (List[int]): The number of units in each layer.
        name (str, optional): The name of the model. Defaults to ''.
    """

    def __init__(self, dim_state, normalizer, n_units, cfgs, *, name='') -> None:
        """Initialize the transition model."""
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = n_units[0] - dim_state
        self.normalizer = normalizer
        self.net = MultiLayerPerceptron(n_units, activation=nn.SiLU)
        self.init_cfgs(cfgs)
        self.max_log_std = nn.Parameter(torch.full([dim_state], 0.5), requires_grad=True)
        self.min_log_std = nn.Parameter(torch.full([dim_state], -10.0), requires_grad=True)
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.name = name
        self.mul_std = self.mul_std  # type: ignore
        self.automatic_optimization = False

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: The configurations.
        """
        self.batch_size = cfgs.batch_size
        self.weight_decay = cfgs.weight_decay
        self.lr = cfgs.lr
        self.mul_std = cfgs.mul_std

    def forward(self, states, actions, det=True):
        """Forward pass of the transition model.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.
            det (bool, optional): Whether to use deterministic output. Defaults to True.
        """
        output = self.net(self.normalizer(states), actions)
        mean, log_std = output.split(self.dim_state, dim=-1)
        if self.mul_std:
            mean = mean * self.normalizer.std
        mean = mean + states
        if det:
            return mean
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return torch.distributions.Normal(mean, log_std.exp())

    def get_loss(self, batch, gp=False):
        """Get the loss of the transition model.

        Args:
            batch (dict[str, torch.Tensor]): The batch data.
            gp (bool, optional): Whether to use gradient penalty. Defaults to False.
        """
        batch['obs'].requires_grad_()
        predictions: torch.distributions.Normal = self(batch['obs'], batch['act'], det=False)
        targets = batch['next_obs']
        loss = (
            -predictions.log_prob(targets).mean()
            + 0.001 * (self.max_log_std - self.min_log_std).mean()
        )
        if gp:
            grad = torch.autograd.grad(loss.sum(), batch['obs'], create_graph=True)[0]
            grad_penalty = (grad.norm(dim=-1) - 1).relu().pow(2).sum()
            loss = loss + 10 * grad_penalty
        return loss

    def configure_optimizers(self):
        """Configure the optimizers for the transition model.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def training_step(self, batch):
        """Training step of the transition model.

        Args:
            batch (dict[str, torch.Tensor]): The batch data.

        Returns:
            dict[str, float]: The training loss.
        """
        loss = self.get_loss(batch, gp=True)
        self.log(f'{self.name}/training_loss', loss.item(), on_step=False, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()  # type: ignore
        self.manual_backward(loss, opt)
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()  # type: ignore

        return {
            'loss': loss.item(),
        }

    def validation_step(self, batch):
        """Validation step of the transition model.

        Args:
            batch (dict[str, torch.Tensor]): The batch data.

        Returns:
            dict[str, float]: The validation loss.
        """
        loss = self.get_loss(batch)
        self.log(f'{self.name}/val_loss', loss.item(), on_step=False, on_epoch=True)
        return {
            'loss': loss.item(),
        }

    def on_epoch_end(self) -> None:
        """Called at the end of an epoch."""
        print('epoch', self.current_epoch, self.device)


class CrabsCore(torch.nn.Module):
    """Core class for CRABS.

    It encapsulates the core process of barrier function.
    For more details, you can refer to the paper: https://arxiv.org/abs/2108.01846

    Args:
        h: The barrier function.
        model: The ensemble model for transition dynamics.
        policy: The policy.
    """

    def __init__(self, h, model: EnsembleModel, policy: ConstraintActorQCritic, cfgs) -> None:
        """Initialize the CRABS core."""
        super().__init__()
        self.h = h
        self.policy = policy
        self.model = model

        self.init_cfgs(cfgs)

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: The configurations.
        """
        self.eps = cfgs.obj.eps
        self.neg_coef = cfgs.obj.neg_coef

    def u(self, states, actions=None):
        """Compute the value of the barrier function.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor, optional): The actions. Defaults to None.
        """
        if actions is None:
            actions = self.policy(states)

        next_states = [self.model.models[idx](states, actions) for idx in self.model.elites]

        all_next_states = torch.stack(next_states)
        all_nh = self.h(all_next_states)
        return all_nh.max(dim=0).values

    def obj_eval(self, s):
        """Short cut for barrier function.

        Args:
            s: The states.

        Returns:
            dict: The results of the barrier function.
        """
        h = self.h(s)
        u = self.u(s)

        eps = self.eps
        obj = u + eps
        mask = (h < 0) & (u + eps > 0)
        return {
            'h': h,
            'u': u,
            's': s,
            'obj': obj,
            'constraint': h,
            'mask': mask,
            'max_obj': (obj * mask).max(),
            'hard_obj': torch.where(h < 0, u + eps, -h - 1000),
        }


class GatedTransitionModel(TransitionModel):
    """Gated transition model for dynamics."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the gated transition model."""
        super().__init__(*args, **kwargs)
        self.gate_net = MultiLayerPerceptron(
            [self.dim_state + self.dim_action, 256, 256, self.dim_state * 2],
            activation=nn.SiLU,
            output_activation=nn.Sigmoid,
        )

    def forward(self, states, actions, det=True):
        """Forward pass of the gated transition model.

        Args:
            states (torch.Tensor): The states.
            actions (torch.Tensor): The actions.
            det (bool, optional): Whether to use deterministic output. Defaults to True.
        """
        nmlz_states = self.normalizer(states)
        reset, update = self.gate_net(nmlz_states, actions).split(self.dim_state, dim=-1)
        output = self.net(nmlz_states * reset, actions)
        mean, log_std = output.split(self.dim_state, dim=-1)
        mean = mean * update + states
        if det:
            return mean
        log_std = self.max_log_std - F.softplus(self.max_log_std - log_std)
        log_std = self.min_log_std + F.softplus(log_std - self.min_log_std)
        return torch.distributions.Normal(mean, log_std.exp())


class BasePolicy(abc.ABC):
    """Base class for policy."""

    @abc.abstractmethod
    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.
        """


class ExplorationPolicy(nn.Module, BasePolicy):
    """Exploration policy for CRABS.

    Args:
        policy (BasePolicy): The policy.
        core (CrabsCore): The CRABS core.
    """

    def __init__(self, policy, core: CrabsCore) -> None:
        """Initialize the exploration policy."""
        super().__init__()
        self.policy = policy
        self.crabs = core
        self.last_h = 0
        self.last_u = 0

    @torch.no_grad()
    def forward(self, states: torch.Tensor):
        """Safe exploration policy.

        Certify the safety of the action by the barrier function.

        Args:
            states (torch.Tensor): The states.
        """
        device = states.device
        assert len(states) == 1
        dist = self.policy(states)

        if isinstance(dist, TanhNormal):
            mean, std = dist.mean, dist.stddev

            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10.0, device=device)
            actions = (
                mean + torch.randn([n, *mean.shape[1:]], device=device) * std * decay[:, None]
            ).tanh()
        else:
            mean = dist
            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10.0, device=device)
            actions = mean + torch.randn([n, *mean.shape[1:]], device=device) * decay[:, None]

        all_u = self.crabs.u(states, actions).detach().cpu().numpy()

        if np.min(all_u) <= 0:
            index = np.min(np.where(all_u <= 0)[0])
            action = actions[index]
        else:
            action = self.crabs.policy(states[0])

        return action[None]

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.forward(obs)

    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The sampled actions.
        """
        return self(states)


class NetPolicy(nn.Module, BasePolicy):
    """Base class for policy."""

    def get_actions(self, states):
        """Sample the actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The sampled actions.
        """
        return self(states).sample()


class DetNetPolicy(NetPolicy):
    """Deterministic policy for CRABS."""

    def get_actions(self, states):
        """Get the deterministic actions.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The deterministic actions.
        """
        return self(states)


class MeanPolicy(DetNetPolicy):
    """Mean policy for CRABS.

    Args:
        policy (NetPolicy): The policy.
    """

    def __init__(self, policy) -> None:
        """Initialize the mean policy."""
        super().__init__()
        self.policy = policy

    def forward(self, states):
        """Forward pass of the mean policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The mean of the policy.
        """
        return self.policy(states).mean

    def step(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Choose the action based on the observation. used in rollout without gradient.

        Args:
            obs (torch.tensor): The observation from environments.
            deterministic (bool, optional): Whether to use deterministic action. Defaults to False.

        Returns:
            The deterministic action if deterministic is True.
            Action with noise other wise.
        """
        with torch.no_grad():
            return self.policy(obs).mean


class AddGaussianNoise(NetPolicy):
    """Add Gaussian noise to the actions.

    Args:
        policy (NetPolicy): The policy.
        mean: The mean of the noise.
        std: The standard deviation of the noise.
    """

    def __init__(self, policy: NetPolicy, mean, std) -> None:
        """Initialize the policy with Gaussian noise."""
        super().__init__()
        self.policy = policy
        self.mean = mean
        self.std = std

    def forward(self, states):
        """Forward pass of the policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The actions.
        """
        actions = self.policy(states)
        if isinstance(actions, TanhNormal):
            return TanhNormal(actions.mean + self.mean, actions.stddev * self.std)
        noises = torch.randn(*actions.shape, device=states.device) * self.std + self.mean
        return actions + noises


class UniformPolicy(NetPolicy):
    """Uniform policy for CRABS.

    Args:
        dim_action (int): The dimension of the action.
    """

    def __init__(self, dim_action) -> None:
        """Initialize the uniform policy."""
        super().__init__()
        self.dim_action = dim_action

    def forward(self, states):
        """Forward pass of the policy.

        Args:
            states (torch.Tensor): The states.

        Returns:
            torch.Tensor: The actions.
        """
        return torch.rand(states.shape[:-1] + (self.dim_action,), device=states.device) * 2 - 1
