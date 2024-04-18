from typing import Callable
import torch
from omnisafe.common.control_barrier_function.models import CrabsCore
from torch.nn.functional import relu, softplus
from torch import nn, optim
import numpy as np
from rich.progress import track
from typing import Callable, Tuple


class Barrier(nn.Module):
    class FLAGS:
        ell_coef = 1.
        barrier_coef = 1

    def __init__(self, net, env_barrier_fn, s0):
        super().__init__()
        self.net = net
        self.env_barrier_fn = env_barrier_fn
        self.s0 = s0
        self.ell = softplus

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.ell(self.net(states) - self.net(self.s0[None])) * self.FLAGS.ell_coef \
               + self.env_barrier_fn(states) * self.FLAGS.barrier_coef - 1

class StateBox:
    INF = 1e10

    def __init__(self, shape, s0, device, expansion=1.5, logger=None):
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
        s = torch.empty(10_000, *self.shape, device=self.device)
        count = 0
        for i in range(1000):
            self.fill_(s)
            inside = torch.where(h(s) < 0.0)[0]
            if len(inside) and (torch.any(s[inside] < self._min) or torch.any(s[inside] > self._max)):
                self.update(s[inside])
                count += 1
            else:
                break

    def update(self, data, logging=True):
        self._max = self._max.maximum(data.max(dim=0).values)
        self._min = self._min.minimum(data.min(dim=0).values)
        self.center = (self._max + self._min) / 2
        self.length = (self._max - self._min) / 2 * self.expansion  # expand the box

    @torch.no_grad()
    def reset(self):
        nn.init.constant_(self._max, -self.INF)
        nn.init.constant_(self._min, +self.INF)
        self.update(self.s0 + 1e-3, logging=False)
        self.update(self.s0 - 1e-3, logging=False)

    @torch.no_grad()
    def fill_(self, s):
        s.data.copy_((torch.rand_like(s) * 2 - 1) * self.length + self.center)

    def decode(self, s):
        return s * self.length + self.center

class SLangevinOptimizer(nn.Module):
    class FLAGS:
        class temperature:
            max = 0.03
            min = 0.03

        class filter:
            top_k = 10000
            pool = False

        n_steps = 1
        method = 'MALA'
        lr = 0.01
        batch_size = 10000
        extend_region = 0.0
        barrier_coef = 0.
        L_neg_coef = 1
        resample = False

        n_proj_iters = 10
        precond = False

    def __init__(self, core: CrabsCore, state_box: StateBox, cfgs, logger):
        super().__init__()
        self.core = core
        self.temperature = self.FLAGS.temperature.max
        self.state_box = state_box

        self._cfgs = cfgs
        self._logger = logger

        self.z = nn.Parameter(torch.zeros(self.FLAGS.batch_size, *state_box.shape, device=self._cfgs.train_cfgs.device), requires_grad=True)
        self.tau = nn.Parameter(torch.full([self.FLAGS.batch_size, 1], 1e-2), requires_grad=False)
        self.alpha = nn.Parameter(torch.full([self.FLAGS.batch_size], 3.0), requires_grad=False)
        self.opt = torch.optim.Adam([self.z])
        self.max_s = torch.zeros(state_box.shape, device=self._cfgs.train_cfgs.device)
        self.min_s = torch.zeros(state_box.shape, device=self._cfgs.train_cfgs.device)

        self.mask = torch.tensor([0], dtype=torch.int64)
        self.n_failure = torch.zeros(self.FLAGS.batch_size, dtype=torch.int64, device=self._cfgs.train_cfgs.device)
        self.n_resampled = 0

        self.adam = torch.optim.Adam([self.z], betas=(0, 0.999), lr=0.001)
        self.since_last_reset = 0
        self.reinit()

    @property
    def s(self):
        return self.state_box.decode(self.z)

    def reinit(self):
        nn.init.uniform_(self.z, -1., 1.)
        nn.init.constant_(self.tau, 0.01)
        nn.init.constant_(self.alpha, 3.0)
        self.since_last_reset = 0

    def set_temperature(self, p):
        max = self.FLAGS.temperature.max
        min = self.FLAGS.temperature.min
        self.temperature = np.exp(np.log(max) * (1 - p) + np.log(min) * p)

    def pdf(self, z):
        s = self.state_box.decode(z)
        result = self.core.obj_eval(s)

        return result['hard_obj'] / self.temperature, result

    def project_back(self, should_print=False):
        for _ in range(self.FLAGS.n_proj_iters):
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
        if len(idx) == 0:
            return
        new_idx = f.softmax(0).multinomial(len(idx), replacement=True)
        self.z[idx] = self.z[new_idx]
        self.tau[idx] = self.tau[new_idx]
        self.n_failure[idx] = 0
        self.n_resampled += len(idx)

    def step(self):
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
        going_out = (a_info['h'] < 0) & (b_info['h'] > 0)


        with torch.no_grad():
            log_p_a_to_b = -w.norm(dim=-1)**2
            log_p_b_to_a = -((a - b - tau * grad_b)**2).sum(dim=-1) / tau[:, 0] / 4

            log_ratio = (f_b + log_p_b_to_a) - (f_a + log_p_a_to_b)

            ratio = log_ratio.clamp(max=0).exp()[:, None]

            sampling = torch.rand_like(ratio) < ratio

            b = torch.where(sampling.squeeze((0,1))[:, None] & (b_info['h'][:, None].squeeze((0,1))[:, None] < 0), b, a)

            new_f_b = torch.where(sampling[:, 0], f_b, f_a)

            self.mask = torch.nonzero(new_f_b >= 0)[:, 0]
            if len(self.mask) == 0:
                self.mask = torch.tensor([0], dtype=torch.int64)

            self.z.set_(b)

            self.tau.mul_(self.FLAGS.lr * (ratio.squeeze()[:, None] - 0.574) + 1)  # .clamp_(max=1.0)
            if self.FLAGS.resample:
                self.n_failure[new_f_b >= -100] = 0
                self.n_failure += 1
                self.resample(new_f_b, torch.nonzero(self.n_failure > 1000)[:, 0])
        return {
            'optimal': a_info['hard_obj'].max().item(),
        }

    @torch.no_grad()
    def debug(self, *, step=0):
        result = self.core.obj_eval(self.s)
        h = result['h']
        hardD_s = result['hard_obj'].max().item()
        inside = (result['constraint'] <= 0).sum().item()
        cut_size = result['mask'].sum().item()

        geo_mean_tau = self.tau.log().mean().exp().item()
        max_tau = self.tau.max().item()

        h_inside = h.cpu().numpy()
        h_inside = h_inside[np.where(result['constraint'].cpu() <= 0)]
        h_dist = np.percentile(h_inside, [25, 50, 75]) if len(h_inside) else []

        self.n_resampled = 0

        return {
            'inside': inside
        }

class BarrierCertOptimizer:
    class FLAGS:
        weight_decay = 1e-4
        lr = 0.0003
        lambda_2 = 'norm'
        locals = {}

    def __init__(self, h: Barrier, obj_eval: Callable[[torch.Tensor], dict], core_ref: CrabsCore, s_opt: SLangevinOptimizer,
                 state_box: StateBox, h_ref: Barrier = None, cfgs = None):
        super().__init__()

        self._cfgs = cfgs

        self.h = h
        self.obj_eval = obj_eval
        self.core_ref = core_ref
        self.s_opt = s_opt
        self.state_box = state_box
        self.h_ref = h_ref
        self.s_opt_sample = SSampleOptimizer(self.obj_eval, self.state_box).to(self._cfgs.device)
        self.s_opt_grad = SGradOptimizer(self.obj_eval, self.state_box).to(self._cfgs.device)

        self.since_last_update = 0
        self.opt = torch.optim.Adam(self.h.parameters(), lr=self.FLAGS.lr, weight_decay=self.FLAGS.weight_decay)

    def step(self):
        for i in range(self._cfgs.opt_s.n_steps):
            self.s_opt.step()
        s = self.s_opt.s.detach().clone().requires_grad_()
        result = self.obj_eval(s)
        mask, obj = result['mask'], result['obj']
        regularization = 0
        if self.h_ref is not None:
            regularization = regularization + (result['h'] - self.h_ref(s)).clamp(min=0.).mean() * 0.001

        if mask.sum() > 0:
            constrained_optimize(obj * mask / mask.sum(), result['constraint'], s, self.opt, reg=regularization)
            self.since_last_update = 0
        else:
            self.since_last_update += 1
        return result

class SSampleOptimizer(nn.Module):
    def __init__(self, obj_eval: Callable[[torch.Tensor], dict], state_box: StateBox, logger=None):
        super().__init__()
        self.obj_eval = obj_eval
        self.s = nn.Parameter(torch.randn(100_000, *state_box.shape), requires_grad=False)
        self.state_box = state_box

        self._logger = logger

    @torch.no_grad()
    def debug(self, *, step):
        self.state_box.fill_(self.s)
        s = self.s
        result = self.obj_eval(s)

        hardD_sample_s = result['hard_obj'].max().item()
        inside = (result['h'] <= 0).sum().item()


class SGradOptimizer(nn.Module):
    def __init__(self, obj_eval: Callable[[torch.Tensor], dict], state_box: StateBox, logger=None):
        super().__init__()
        self.obj_eval = obj_eval
        self.z = nn.Parameter(torch.randn(10000, *state_box.shape), requires_grad=True)
        self.opt = torch.optim.Adam([self.z], lr=1e-3)
        self.state_box = state_box

        self._logger = logger

    @property
    def s(self):
        return self.state_box.decode(self.z)

    def step(self):
        result = self.obj_eval(self.s)
        obj = result['hard_obj']
        loss = (-obj).mean()

        self.opt.zero_grad()
        loss.mean().backward()
        self.opt.step()
        return loss

    @torch.no_grad()
    def reinit(self):
        nn.init.uniform_(self.z, -1., 1.)

    def debug(self, *, step):
        result = self.obj_eval(self.s)
        hardD = result['hard_obj']
        h = result['constraint']
        u = result['obj']
        idx = hardD.argmax()
        max_obj = hardD.max().item()
        if max_obj > 0:
            prefix = 'Warning: '
        else:
            prefix = 'Debug: '

        return {
            'optimal': hardD.max().item(),
        }

class PolicyAdvTraining:

    def __init__(self, policy, s_opt, obj_eval, cfgs):
        self.policy = policy
        self.s_opt = s_opt
        self.obj_eval = obj_eval

        self._cfgs = cfgs

        self.weight_decay = 1e-4
        self.lr = 0.0003

        self.count = 0.0
        self.opt = torch.optim.Adam(policy.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self, freq):
        for i in range(self._cfgs.opt_s.n_steps):
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

    def __init__(self, h: Barrier, obj_eval: Callable[[torch.Tensor], dict], core_ref: CrabsCore, s_opt: SLangevinOptimizer,
                 state_box: StateBox, h_ref: Barrier = None, cfgs=None, logger=None):
        super().__init__()
        self.h = h
        self.obj_eval = obj_eval
        self.core_ref = core_ref
        self.s_opt = s_opt
        self.state_box = state_box
        self.h_ref = h_ref

        self._cfgs = cfgs
        self._logger = logger

        self.s_opt_sample = SSampleOptimizer(self.obj_eval, self.state_box).to(self._cfgs.train_cfgs.device)
        self.s_opt_grad = SGradOptimizer(self.obj_eval, self.state_box).to(self._cfgs.train_cfgs.device)

        self.weight_decay = 1e-4
        self.lr = 0.0003
        self.lambda_2 = 'norm'
        self.locals = {}

        self.since_last_update = 0
        self.opt = torch.optim.Adam(self.h.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def step(self):
        for i in range(self._cfgs.opt_s.n_steps):
            self.s_opt.step()
        s = self.s_opt.s.detach().clone().requires_grad_()
        result = self.obj_eval(s)
        mask, obj = result['mask'], result['obj']
        regularization = 0
        if self.h_ref is not None:
            regularization = regularization + (result['h'] - self.h_ref(s)).clamp(min=0.).mean() * 0.001

        if mask.sum() > 0:
            constrained_optimize(obj * mask / mask.sum(), result['constraint'], s, self.opt, reg=regularization)
            self.since_last_update = 0
        else:
            self.since_last_update += 1
        return result

    def debug(self, *, step=0):
        self.s_opt.debug(step=step)


    def train(self) -> Tuple[bool, float]:
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
        # don't tune state box
        self.state_box.reset()
        for i in range(self._cfgs.n_pretrain_s_iters):
            if i % 1_000 == 0:
                self.s_opt.debug(step=i)
            self.s_opt.step()

        self.h_ref = None
        for t in range(self._cfgs.n_iters):
            if t % 1_000 == 0:
                self.check_by_sample(step=t)
                self.s_opt.debug(step=t)

                result = self.obj_eval(self.s_opt.s)
                hardD = result['hard_obj']

            if t % 50_000 == 0 and t > 0:
                self.check_by_grad()

            self.step()

            if self.since_last_update > 2000 and self.s_opt.since_last_reset > 5000:
                self.state_box.reset()
                self.state_box.find_box(self.h)
                self.s_opt.reinit()

    def check_by_sample(self, *, step=0):
        self.s_opt_sample.debug(step=step)

    def check_by_grad(self):
        self.s_opt_grad.reinit()
        for i in range(10001): # debug, change as 10001
            if i % 1000 == 0:
                self.s_opt_grad.debug(step=0)
            self.s_opt_grad.step()
