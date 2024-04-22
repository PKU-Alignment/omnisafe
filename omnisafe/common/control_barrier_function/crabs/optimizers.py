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
"""Optimizers for CRABS."""
# pylint: disable=all
from typing import Callable, Tuple

import numpy as np
import torch
from rich.progress import track
from torch import nn
from torch.nn.functional import relu, softplus

from omnisafe.common.control_barrier_function.crabs.models import CrabsCore


class Barrier(nn.Module):
    """Barrier function for the environment.

    This is corresponding to the function h(x) in the paper.

    Args:
        net (nn.Module): Neural network that represents the barrier function.
        env_barrier_fn (Callable): Barrier function for the environment.
        s0 (torch.Tensor): Initial state.
    """

    def __init__(self, net, env_barrier_fn, s0, cfgs) -> None:
        """Initialize the barrier function."""
        super().__init__()
        self.net = net
        self.env_barrier_fn = env_barrier_fn
        self.s0 = s0
        self.ell = softplus

        self.ell_coef = cfgs.ell_coef
        self.barrier_coef = cfgs.barrier_coef

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass of the barrier function.

        Args:
            states (torch.Tensor): States to evaluate the barrier function.

        Returns:
            torch.Tensor: Barrier function values.
        """
        return (
            self.ell(self.net(states) - self.net(self.s0[None])) * self.ell_coef
            + self.env_barrier_fn(states) * self.barrier_coef
            - 1
        )


class StateBox:
    """State box for the environment.

    Args:
        shape (Tuple): Shape of the state box.
        s0 (torch.Tensor): Initial state.
        device (torch.device): Device to run the state box.
        expansion (float, optional): Expansion factor for the state box. Defaults to 1.5.
        logger ([type], optional): Logger for the state box. Defaults to None.
    """

    INF = 1e10

    def __init__(self, shape, s0, device, expansion=1.5, logger=None) -> None:
        """Initialize the state box."""
        self._max = torch.full(shape, -self.INF, device=device)
        self._min = torch.full(shape, +self.INF, device=device)
        self.center = None
        self.length = None
        self.expansion = expansion
        self.device = device
        self.s0 = s0
        self.shape = shape

        self._logger = logger

    @torch.no_grad()
    def find_box(self, h):
        """Find the state box.

        Args:
            h: Barrier function.
        """
        s = torch.empty(10_000, *self.shape, device=self.device)
        count = 0
        for _i in range(1000):
            self.fill_(s)
            inside = torch.where(h(s) < 0.0)[0]
            if len(inside) and (
                torch.any(s[inside] < self._min) or torch.any(s[inside] > self._max)
            ):
                self.update(s[inside])
                count += 1
            else:
                break

    def update(self, data, logging=True):
        """Update the state box.

        Args:
            data (torch.Tensor): Data to update the state box.
            logging (bool, optional): Whether to log the state box. Defaults to True.
        """
        self._max = self._max.maximum(data.max(dim=0).values)
        self._min = self._min.minimum(data.min(dim=0).values)
        self.center = (self._max + self._min) / 2  # type: ignore
        self.length = (self._max - self._min) / 2 * self.expansion  # expand the box

    @torch.no_grad()
    def reset(self):
        """Reset the state box."""
        nn.init.constant_(self._max, -self.INF)
        nn.init.constant_(self._min, +self.INF)
        self.update(self.s0 + 1e-3, logging=False)
        self.update(self.s0 - 1e-3, logging=False)

    @torch.no_grad()
    def fill_(self, s):
        """Fill the state box.

        Args:
            s (torch.Tensor): State tensor to fill.
        """
        s.data.copy_((torch.rand_like(s) * 2 - 1) * self.length + self.center)

    def decode(self, s):
        """Decode the state via the state box.

        Args:
            s (torch.Tensor): State tensor to decode.

        Returns:
            torch.Tensor: Decoded state.
        """
        return s * self.length + self.center


class SLangevinOptimizer(nn.Module):
    """Stochastic Langevin optimizer for the s*.

    This class is used to optimize the s* in the paper.

    Args:
        core (CrabsCore): Core model for the optimization.
        state_box (StateBox): State box for the optimization.
        cfgs: Configuration for the optimization.
        logger: Logger for the optimization.
    """

    def __init__(self, core: CrabsCore, state_box: StateBox, device, cfgs, logger) -> None:
        """Initialize the optimizer."""
        super().__init__()
        self.core = core
        self.state_box = state_box

        self._cfgs = cfgs
        self._logger = logger
        self.init_cfgs(cfgs)

        self.temperature = self.temperature.max  # type: ignore
        self.z = nn.Parameter(
            torch.zeros(
                self.batch_size,  # type: ignore
                *state_box.shape,
                device=device,
            ),
            requires_grad=True,
        )
        self.tau = nn.Parameter(torch.full([self.batch_size, 1], 1e-2), requires_grad=False)  # type: ignore
        self.alpha = nn.Parameter(torch.full([self.batch_size], 3.0), requires_grad=False)  # type: ignore
        self.opt = torch.optim.Adam([self.z])
        self.max_s = torch.zeros(state_box.shape, device=device)
        self.min_s = torch.zeros(state_box.shape, device=device)

        self.mask = torch.tensor([0], dtype=torch.int64)
        self.n_failure = torch.zeros(
            self.batch_size,  # type: ignore
            dtype=torch.int64,
            device=device,
        )
        self.n_resampled = 0

        self.adam = torch.optim.Adam([self.z], betas=(0, 0.999), lr=0.001)
        self.since_last_reset = 0
        self.reinit()

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: Configuration for the optimization.
        """
        self.temperature = cfgs.temperature

        self.filter = cfgs.filter

        self.n_steps = cfgs.n_steps
        self.method = cfgs.method
        self.lr = cfgs.lr
        self.batch_size = cfgs.batch_size
        self.extend_region = cfgs.extend_region
        self.barrier_coef = cfgs.barrier_coef
        self.L_neg_coef = cfgs.L_neg_coef
        self.is_resample = cfgs.resample

        self.n_proj_iters = cfgs.n_proj_iters
        self.precond = cfgs.precond

    @property
    def s(self):
        """Decoded state from the state box.

        Returns:
            torch.Tensor: Decoded state.
        """
        return self.state_box.decode(self.z)

    def reinit(self):
        """Reinitialize the optimizer."""
        nn.init.uniform_(self.z, -1.0, 1.0)
        nn.init.constant_(self.tau, 0.01)
        nn.init.constant_(self.alpha, 3.0)
        self.since_last_reset = 0

    def set_temperature(self, p):
        """Set the temperature for the optimizer.

        Args:
            p (float): Temperature parameter.
        """
        max = self.temperature.max
        min = self.temperature.min
        self.temperature = np.exp(np.log(max) * (1 - p) + np.log(min) * p)

    def pdf(self, z):
        """Probability density function for the optimizer.

        Args:
            z (torch.Tensor): State tensor.

        Returns:
            Tuple[torch.Tensor, dict]: Probability density function and the result.
        """
        s = self.state_box.decode(z)
        result = self.core.obj_eval(s)

        return result['hard_obj'] / self.temperature, result

    def project_back(self):
        """Use gradient descent to project s back to the set C_h."""
        for _ in range(self.n_proj_iters):
            with torch.enable_grad():
                h = self.core.h(self.s)
                loss = relu(h - 0.03)
                if (h > 0.03).sum() < 1000:
                    break
                self.adam.zero_grad()
                loss.sum().backward()
                self.adam.step()

    @torch.no_grad()
    def resample(self, f: torch.Tensor, idx):
        """Resample the states.

        Args:
            f (torch.Tensor): Probability density function.
            idx: Index of the states to resample.
        """
        if len(idx) == 0:
            return
        new_idx = f.softmax(0).multinomial(len(idx), replacement=True)
        self.z[idx] = self.z[new_idx]
        self.tau[idx] = self.tau[new_idx]
        self.n_failure[idx] = 0
        self.n_resampled += len(idx)

    def step(self):
        """One step of the optimizer."""
        self.since_last_reset += 1
        self.project_back()
        tau = self.tau
        a = self.z

        f_a, a_info = self.pdf(a)
        grad_a = torch.autograd.grad(f_a.sum(), a)[0]

        w = torch.randn_like(a)
        b = a + tau * grad_a + (tau * 2).sqrt() * w
        b = b.detach().requires_grad_()
        f_b, b_info = self.pdf(b)

        grad_b = torch.autograd.grad(f_b.sum(), b)[0]
        (a_info['h'] < 0) & (b_info['h'] > 0)

        with torch.no_grad():
            log_p_a_to_b = -w.norm(dim=-1) ** 2
            log_p_b_to_a = -((a - b - tau * grad_b) ** 2).sum(dim=-1) / tau[:, 0] / 4

            log_ratio = (f_b + log_p_b_to_a) - (f_a + log_p_a_to_b)

            ratio = log_ratio.clamp(max=0).exp()[:, None]

            sampling = torch.rand_like(ratio) < ratio

            b = torch.where(
                sampling.squeeze((0, 1))[:, None]
                & (b_info['h'][:, None].squeeze((0, 1))[:, None] < 0),
                b,
                a,
            )

            new_f_b = torch.where(sampling[:, 0], f_b, f_a)

            self.mask = torch.nonzero(new_f_b >= 0)[:, 0]
            if len(self.mask) == 0:
                self.mask = torch.tensor([0], dtype=torch.int64)

            self.z.set_(b)  # type: ignore

            self.tau.mul_(
                self.lr * (ratio.squeeze()[:, None] - 0.574) + 1,
            )
            if self.is_resample:
                self.n_failure[new_f_b >= -100] = 0
                self.n_failure += 1
                self.resample(new_f_b, torch.nonzero(self.n_failure > 1000)[:, 0])
        return {
            'optimal': a_info['hard_obj'].max().item(),
        }

    @torch.no_grad()
    def debug(self, *, step=0):
        """Debug."""
        result = self.core.obj_eval(self.s)
        h = result['h']
        result['hard_obj'].max().item()
        inside = (result['constraint'] <= 0).sum().item()
        result['mask'].sum().item()

        self.tau.log().mean().exp().item()
        self.tau.max().item()

        h_inside = h.cpu().numpy()
        h_inside = h_inside[np.where(result['constraint'].cpu() <= 0)]
        np.percentile(h_inside, [25, 50, 75]) if len(h_inside) else []

        self.n_resampled = 0

        return {
            'inside': inside,
        }


class SSampleOptimizer(nn.Module):
    """Sample optimizer for the s*.

    Args:
        obj_eval (Callable): Objective evaluation function.
        state_box (StateBox): State box.
        logger: Logger for the optimizer.
    """

    def __init__(
        self,
        obj_eval: Callable[[torch.Tensor], dict],
        state_box: StateBox,
        logger=None,
    ) -> None:
        """Initialize the optimizer."""
        super().__init__()
        self.obj_eval = obj_eval
        self.s = nn.Parameter(torch.randn(100_000, *state_box.shape), requires_grad=False)
        self.state_box = state_box

        self._logger = logger

    @torch.no_grad()
    def debug(self, *, step):
        """Debug."""
        self.state_box.fill_(self.s)
        s = self.s
        result = self.obj_eval(s)

        result['hard_obj'].max().item()
        (result['h'] <= 0).sum().item()


class SGradOptimizer(nn.Module):
    """Gradient optimizer for the s*.

    Args:
        obj_eval (Callable): Objective evaluation function.
        state_box (StateBox): State box.
        logger: Logger for the optimizer.
    """

    def __init__(
        self,
        obj_eval: Callable[[torch.Tensor], dict],
        state_box: StateBox,
        logger=None,
    ) -> None:
        """Initialize the optimizer."""
        super().__init__()
        self.obj_eval = obj_eval
        self.z = nn.Parameter(torch.randn(10000, *state_box.shape), requires_grad=True)
        self.opt = torch.optim.Adam([self.z], lr=1e-3)
        self.state_box = state_box

        self._logger = logger

    @property
    def s(self):
        """Decoded state from the state box.

        Returns:
            torch.Tensor: Decoded state.
        """
        return self.state_box.decode(self.z)

    def step(self):
        """One step of the optimizer.

        Returns:
            torch.Tensor: Loss.
        """
        result = self.obj_eval(self.s)
        obj = result['hard_obj']
        loss = (-obj).mean()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss

    @torch.no_grad()
    def reinit(self):
        """Reinitialize the optimizer."""
        nn.init.uniform_(self.z, -1.0, 1.0)

    def debug(self, *, step):
        """Debug."""
        result = self.obj_eval(self.s)
        hardD = result['hard_obj']
        result['constraint']
        result['obj']
        hardD.argmax()
        hardD.max().item()

        return {
            'optimal': hardD.max().item(),
        }


class PolicyAdvTraining:
    """Policy adversarial training.

    Args:
        policy (nn.Module): Policy model.
        s_opt (SLangevinOptimizer): Stochastic Langevin optimizer.
        obj_eval (Callable): Objective evaluation function.
        cfgs: Configuration for the optimizer.
    """

    def __init__(self, policy, s_opt, obj_eval, cfgs) -> None:
        """Initialize the optimizer."""
        self.policy = policy
        self.s_opt = s_opt
        self.obj_eval = obj_eval

        self._cfgs = cfgs

        self.weight_decay = 1e-4
        self.lr = 0.0003

        self.count = 0.0
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self, freq):
        """One step of the optimizer."""
        for _i in range(self._cfgs.opt_s.n_steps):
            self.s_opt.step()

        self.count += 1
        while self.count >= freq:
            self.count -= freq
            result = self.obj_eval(self.s_opt.s)
            mask = result['mask']

            if mask.any():
                self.opt.zero_grad()
                loss = (result['obj'] * mask).sum() / mask.sum()
                loss.backward()
                self.opt.step()


@torch.enable_grad()
def constrained_optimize(fx, gx, x, opt, reg=0.0):  # \grad_y [max_{x: g_y(x) <= 0} f(x)]
    """Constrained optimization.

    Args:
        fx (torch.Tensor): Function to optimize.
        gx (torch.Tensor): Constraint function.
        x (torch.Tensor): State tensor.
        opt: Optimizer.
        reg (float, optional): Regularization term. Defaults to 0.0.
    """
    sum_fx = fx.sum()
    sum_gx = gx.sum()
    with torch.no_grad():
        df_x = torch.autograd.grad(sum_fx, x, retain_graph=True)[0]
        dg_x = torch.autograd.grad(sum_gx, x, retain_graph=True)[0]
        lambda_ = df_x.norm(dim=-1) / dg_x.norm(dim=-1).clamp(min=1e-6)

    opt.zero_grad()
    (fx - gx * lambda_ + reg).sum().backward()
    opt.step()

    return {'df': df_x, 'dg': dg_x}


class BarrierCertOptimizer:
    """Barrier certificate optimizer.

    Args:
        h (Barrier): Barrier function.
        obj_eval (Callable): Objective evaluation function.
        core_ref (CrabsCore): Core model reference.
        s_opt (SLangevinOptimizer): Stochastic Langevin optimizer.
        state_box (StateBox): State box.
        h_ref (Barrier, optional): Reference barrier function. Defaults to None.
        cfgs: Configuration for the optimizer.
        logger: Logger for the optimizer.
    """

    def __init__(
        self,
        h: Barrier,
        obj_eval: Callable[[torch.Tensor], dict],
        core_ref: CrabsCore,
        s_opt: SLangevinOptimizer,
        state_box: StateBox,
        h_ref: Barrier = None,  # type: ignore
        cfgs=None,
        logger=None,
    ) -> None:
        """Initialize the optimizer."""
        super().__init__()
        self.h = h
        self.obj_eval = obj_eval
        self.core_ref = core_ref
        self.s_opt = s_opt
        self.state_box = state_box
        self.h_ref = h_ref

        self._cfgs = cfgs
        self._logger = logger

        self.s_opt_sample = SSampleOptimizer(self.obj_eval, self.state_box).to(
            self._cfgs.train_cfgs.device,
        )
        self.s_opt_grad = SGradOptimizer(self.obj_eval, self.state_box).to(
            self._cfgs.train_cfgs.device,
        )

        self.init_cfgs(cfgs.opt_h)

        self.since_last_update = 0
        self.opt = torch.optim.Adam(self.h.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def init_cfgs(self, cfgs):
        """Initialize the configuration.

        Args:
            cfgs: Configuration for the optimizer.
        """
        self.weight_decay = cfgs.weight_decay
        self.lr = cfgs.lr
        self.lambda_2 = cfgs.lambda_2
        self.locals = cfgs.locals
        self.n_iters = cfgs.n_iters

    def step(self):
        """One step of the optimizer."""
        for _i in range(self._cfgs.opt_s.n_steps):
            self.s_opt.step()
        s = self.s_opt.s.detach().clone().requires_grad_()
        result = self.obj_eval(s)
        mask, obj = result['mask'], result['obj']
        regularization = 0
        if self.h_ref is not None:
            regularization = (
                regularization + (result['h'] - self.h_ref(s)).clamp(min=0.0).mean() * 0.001
            )

        if mask.sum() > 0:
            constrained_optimize(
                obj * mask / mask.sum(),
                result['constraint'],
                s,
                self.opt,
                reg=regularization,
            )
            self.since_last_update = 0
        else:
            self.since_last_update += 1
        return result

    def debug(self, *, step=0):
        """Debug."""
        self.s_opt.debug(step=step)

    def train(self) -> Tuple[bool, float]:
        """Train the barrier certificate.

        Returns:
            Tuple[bool, float]: Whether the training is successful and how to change policy adversarial training frequency.
        """
        for _ in track(range(2000), description='Optimizing s...'):
            self.s_opt.step()

        h_status = 'training'
        self.since_last_update = 0
        for t in track(range(20000), description='Training h...'):
            result = self.step()
            if result['mask'].sum() > 0.0:
                h_status = 'training'

            if h_status == 'training' and self.since_last_update >= 1000:
                self.state_box.find_box(self.core_ref.h)
                self.s_opt.reinit()
                h_status = 'observation-period'

            if h_status == 'observation-period' and self.since_last_update == 5_000:
                if t == 4_999:  # policy is too conservative, reduce safe constraint
                    return True, 2.0
                else:
                    return True, 1.2
        return False, 0.5

    def pretrain(self):
        """Pretrain the barrier certificate."""
        self.state_box.reset()
        for i in range(self._cfgs.n_pretrain_s_iters):
            if i % 1_000 == 0:
                self.s_opt.debug(step=i)
            self.s_opt.step()

        self.h_ref = None  # type: ignore
        for t in range(self.n_iters):
            if t % 1_000 == 0:
                self.check_by_sample(step=t)
                self.s_opt.debug(step=t)

                result = self.obj_eval(self.s_opt.s)
                result['hard_obj']

            if t % 50_000 == 0 and t > 0:
                self.check_by_grad()

            self.step()

            if self.since_last_update > 2000 and self.s_opt.since_last_reset > 5000:
                self.state_box.reset()
                self.state_box.find_box(self.h)
                self.s_opt.reinit()

    def check_by_sample(self, *, step=0):
        """Check if the barrier function is correct by sampling."""
        self.s_opt_sample.debug(step=step)

    def check_by_grad(self):
        """Check if the barrier function is correct by gradient."""
        self.s_opt_grad.reinit()
        for i in range(10001):
            if i % 1000 == 0:
                self.s_opt_grad.debug(step=0)
            self.s_opt_grad.step()
