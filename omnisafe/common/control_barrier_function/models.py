import torch
import torch.nn as nn
import numpy as np
from typing import List
import pytorch_lightning as pl
import abc
from omnisafe.utils.math import TanhNormal

class EnsembleModel(pl.LightningModule):
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.n_models = len(models)
        self.n_elites = self.n_models
        self.elites = []
        self.recompute_elites()
        self.automatic_optimization = False

    def recompute_elites(self):
        self.elites = list(range(len(self.models)))

    def forward(self, states, actions):
        n = len(states)

        perm = np.random.permutation(n)
        inv_perm = np.argsort(perm)

        next_states = []
        for i, (model_idx, indices) in enumerate(zip(self.elites, np.array_split(perm, len(self.elites)))):
            next_states.append(self.models[model_idx](states[indices], actions[indices]))
        return torch.cat(next_states, dim=0)[inv_perm]

    def get_nlls(self, states, actions, next_states):
        ret = []
        for model in self.models:
            distribution = model(states, actions, det=False)
            nll = -distribution.log_prob(next_states).mean().item()
            ret.append(nll)
        return ret

    def training_step(self, batch, batch_idx):
        total_loss = 0
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch, gp=False)
            total_loss = total_loss + loss
            self.log(f'model/{i}/training_loss', loss.item())

        opt = self.optimizers()
        opt.zero_grad()

        self.manual_backward(total_loss)
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()

    def validation_step(self, batch, batch_idx):
        for i, model in enumerate(self.models):
            loss = model.get_loss(batch)
            self.log(f'model/{i}/val_loss', loss.item(), on_step=False)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)


import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np


class MultiLayerPerceptron(nn.Sequential):
    def __init__(self, n_units, activation=nn.ReLU, auto_squeeze=True, output_activation=None):
        layers = []
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
        inputs = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)

        outputs = inputs
        for layer in self:
            outputs = layer(outputs)

        if self._auto_squeeze and outputs.shape[-1] == 1:
            outputs = outputs.squeeze(-1)
        return outputs

    def copy(self):
        return MultiLayerPerceptron(self._n_units, self._activation[0], self._auto_squeeze)

    def extra_repr(self):
        return f'activation = {self._activation}, # units = {self._n_units}, squeeze = {self._auto_squeeze}'


def mixup(batch, alpha=0.2):
    lambda_ = np.random.beta(alpha, alpha)
    batch_size = batch['state'].size(0)
    perm = torch.randperm(batch_size)
    return {
        'state': batch['state'] * lambda_ + batch['state'][perm] * lambda_,
        'action': batch['action'] * lambda_ + batch['action'][perm] * lambda_,
        'next_state': batch['next_state'] * lambda_ + batch['next_state'][perm] * lambda_,
    }

class TransitionModel(pl.LightningModule):

    class FLAGS:
        batch_size = 256
        weight_decay = 0.000075
        lr = 0.001
        mul_std = 0

    def __init__(self, dim_state, normalizer, n_units, *, name=''):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = n_units[0] - dim_state
        self.normalizer = normalizer
        self.net = MultiLayerPerceptron(n_units, activation=nn.SiLU)
        self.max_log_std = nn.Parameter(torch.full([dim_state], 0.5), requires_grad=True)
        self.min_log_std = nn.Parameter(torch.full([dim_state], -10.), requires_grad=True)
        self.training_loss = 0.
        self.val_loss = 0.
        self.name = name
        self.mul_std = self.FLAGS.mul_std
        self.automatic_optimization = False

    def forward(self, states, actions, det=True):
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
        # batch = mixup(batch)
        batch['obs'].requires_grad_()
        predictions: torch.distributions.Normal = self(batch['obs'], batch['act'], det=False)
        targets = batch['next_obs']
        loss = -predictions.log_prob(targets).mean() + 0.001 * (self.max_log_std - self.min_log_std).mean()
        if gp:
            grad = torch.autograd.grad(loss.sum(), batch['obs'], create_graph=True)[0]
            grad_penalty = (grad.norm(dim=-1) - 1).relu().pow(2).sum()
            loss = loss + 10 * grad_penalty
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch, gp=True)
        self.log(f'{self.name}/training_loss', loss.item(), on_step=False, on_epoch=True)

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss, opt)
        nn.utils.clip_grad_norm_(self.parameters(), 10)
        opt.step()

        return {
            'loss': loss.item(),
        }

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.log(f'{self.name}/val_loss', loss.item(), on_step=False, on_epoch=True)
        return {
            'loss': loss.item(),
        }

    def on_epoch_end(self) -> None:
        print('epoch', self.current_epoch, self.device)

    def test_stability(self, state, policy, horizon):
        states = [state]
        for i in range(horizon):
            action = policy(state)
            state = self(state, action)
            states.append(state)
        states = torch.stack(states)
        breakpoint()
        print(states.norm(dim=-1)[::(horizon - 1) // 10])


class CrabsCore(torch.nn.Module):
    class FLAGS:
        class obj:
            eps = 0.01
            neg_coef = 1.0

    def __init__(self, h, model, policy):
        super().__init__()
        self.h = h
        self.policy = policy
        self.model = model

    def u(self, states, actions=None):
        # from time import time as t
        if actions is None:
            actions = self.policy(states)

        next_states = [self.model.models[idx](states, actions) for idx in self.model.elites]

        all_next_states = torch.stack(next_states)
        all_nh = self.h(all_next_states)
        nh = all_nh.max(dim=0).values
        return nh

    def obj_eval(self, s):
        h = self.h(s)
        u = self.u(s)

        # can't be 1e30: otherwise 100 + 1e30 = 1e30
        eps = self.FLAGS.obj.eps
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
            'hard_obj': torch.where(h < 0, u + eps, -h - 1000)
        }


class GatedTransitionModel(TransitionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_net = MultiLayerPerceptron([self.dim_state + self.dim_action, 256, 256, self.dim_state * 2], activation=nn.SiLU,
                                output_activation=nn.Sigmoid)

    def forward(self, states, actions, det=True):
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
    @abc.abstractmethod
    def get_actions(self, states):
        pass

    def reset(self, indices=None):
        pass

class ExplorationPolicy(nn.Module, BasePolicy):
    def __init__(self, policy, core: CrabsCore):
        super().__init__()
        self.policy = policy
        self.crabs = core
        self.last_h = 0
        self.last_u = 0

    @torch.no_grad()
    def forward(self, states: torch.Tensor):
        # from time import time as t
        # s1 = t()
        device = states.device
        assert len(states) == 1
        dist = self.policy(states)

        if isinstance(dist, TanhNormal):
            mean, std = dist.mean, dist.stddev

            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10., device=device)
            actions = (mean + torch.randn([n, *mean.shape[1:]], device=device) * std * decay[:, None]).tanh()
        else:
            mean = dist
            n = 100
            states = states.repeat([n, 1])
            decay = torch.logspace(0, -3, n, base=10., device=device)
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
        return self(states)

class NetPolicy(nn.Module, BasePolicy):
    def get_actions(self, states):
        return self(states).sample()


class DetNetPolicy(NetPolicy):
    def get_actions(self, states):
        return self(states)
    

class MeanPolicy(DetNetPolicy):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, states):
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
    def __init__(self, policy: NetPolicy, mean, std):
        super().__init__()
        self.policy = policy
        self.mean = mean
        self.std = std

    def forward(self, states):
        actions = self.policy(states)
        if isinstance(actions, TanhNormal):
            return TanhNormal(actions.mean + self.mean, actions.stddev * self.std)
        noises = torch.randn(*actions.shape, device=states.device) * self.std + self.mean
        return actions + noises

class UniformPolicy(NetPolicy):
    def __init__(self, dim_action):
        super().__init__()
        self.dim_action = dim_action

    def forward(self, states):
        return torch.rand(states.shape[:-1] + (self.dim_action,), device=states.device) * 2 - 1