"""Optional PyTorch backend (CUDA, or CPU for tests) that trains PolicyNet / ValueNet with the NumPy trainers' semantics.

    python -m training.train_policy --model general --backend torch --device cuda --out runs/policy_general
    python -m training.train_value  --model general --backend torch --device cuda --out runs/value_general

What is reproduced, operation by operation (fusion/nn.py, fusion/models.py):

  - initialization: the NumPy model is built with the run's seed and its arrays are copied, so both backends
    start from the same weights;
  - clipped ReLU on [0, 1] with gradient 1 only for 0 < z < 1 (torch.clamp's gradient includes the endpoints);
  - masked, label-smoothed softmax cross-entropy and the weighted BCE with the NumPy formulas, including the
    gradient w (p - target) for ignored labels, empty masks and labels outside the mask (custom autograd);
  - losses as weighted sums divided by the minibatch weight sum, with the act/target head weights;
  - Adam with coupled L2 decay on matrices only (not torch.optim.Adam, which also decays biases);
  - minibatch order: the same rng.permutation per epoch as training.common.batches, whether the training rows
    are resident on the device or streamed in windows (window sizes are rounded to whole minibatches, so the
    minibatch sequence is identical); counts stay globally mixed in every minibatch;
  - float32 throughout; TF32 matmuls are off unless --tf32 (they change rounding).

Per-epoch validation runs on the device; calibration, the calibrated evaluation, checkpoints and export run
on the NumPy model the trained weights are copied back into, so artifacts are produced exactly as before.
Parity with the NumPy backend is checked by tests/test_torch_backend.py (float tolerances: summation order
differs between BLAS and CUDA; integer export parity is unchanged).
"""
from __future__ import annotations

import threading
import time

import numpy as np
import torch

from .models import JOINT, TARGET_COLS, PolicyNet, ValueNet
from .schema import Layout


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda: PyTorch has no CUDA here (install a CUDA build of PyTorch, see training/specs/ENVIRONMENT.md)")
    return dev


def configure(dev: torch.device, tf32: bool) -> dict:
    """Matmul precision policy; returns the environment record for the run config."""
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    info = {"backend": "torch", "torch": torch.__version__, "device": str(dev), "tf32": tf32}
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        info |= {"gpu": p.name, "gpu_memory_gb": round(p.total_memory / 2 ** 30, 2), "cuda": torch.version.cuda}
    return info


class _ClippedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        ctx.save_for_backward(z)
        return z.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g):
        (z,) = ctx.saved_tensors
        return g * ((z > 0.0) & (z < 1.0)).to(g.dtype)


class _SoftmaxXent(torch.autograd.Function):
    """fusion.nn.softmax_xent: returns the weighted loss sum; backward is w (p - target) times the incoming scale."""

    @staticmethod
    def forward(ctx, logits, mask, label, weight, smoothing: float):
        neg = torch.tensor(float("-inf"), dtype=logits.dtype, device=logits.device)
        z = torch.where(mask, logits, neg)
        mx = z.amax(-1, keepdim=True)
        mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
        e = torch.where(mask, torch.exp(z - mx), torch.zeros_like(z))
        lse = torch.log(e.sum(-1, keepdim=True)) + mx
        logp = torch.where(mask, z - lse, neg)
        p = torch.where(mask, torch.exp(logp), torch.zeros_like(z))
        ok = label >= 0
        # one-hot without torch.nonzero (which would synchronize the device every minibatch)
        target = torch.nn.functional.one_hot(label.clamp(min=0), p.shape[-1]).to(p.dtype) * ok[:, None].to(p.dtype)
        if smoothing:
            legal = mask.sum(-1, keepdim=True).clamp(min=1).to(p.dtype)
            target = torch.where(mask, (1 - smoothing) * target + smoothing / legal, torch.zeros_like(target))
        w = torch.where(ok, weight, torch.zeros_like(weight))[:, None]
        safe_logp = torch.where(mask, logp, torch.zeros_like(logp))
        loss = -(w * torch.where(mask, target * safe_logp, torch.zeros_like(target))).sum()
        ctx.save_for_backward(w * (p - target))
        return loss

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return g * grad, None, None, None, None


class _BCE(torch.autograd.Function):
    """fusion.nn.bce_with_logits: weighted loss sum; backward weight (sigmoid(z) - y)."""

    @staticmethod
    def forward(ctx, z, y, weight):
        loss = torch.clamp(z, min=0) - z * y + torch.log1p(torch.exp(-z.abs()))
        ctx.save_for_backward(weight * (1.0 / (1.0 + torch.exp(-z)) - y))
        return (weight * loss).sum()

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return g * grad, None, None


class TorchMLP:
    """A device copy of a fusion.nn.MLP: same layer order, parameters W0.., b0.."""

    def __init__(self, mlp, dev):
        self.ref = mlp
        self.W = [torch.tensor(w, device=dev, requires_grad=True) for w in mlp.W]
        self.b = [torch.tensor(b, device=dev, requires_grad=True) for b in mlp.b]

    @property
    def params(self):
        return self.W + self.b

    def forward(self, x):
        lead = x.shape[:-1]
        h = x.reshape(-1, x.shape[-1])
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w.T + b
            h = _ClippedReLU.apply(z) if i < len(self.W) - 1 else z
        return h.reshape(*lead, h.shape[-1])

    def to_ref(self):
        for i in range(len(self.W)):
            self.ref.W[i][...] = self.W[i].detach().cpu().numpy()
            self.ref.b[i][...] = self.b[i].detach().cpu().numpy()

    def from_ref(self):
        with torch.no_grad():
            for i in range(len(self.W)):
                self.W[i].copy_(torch.from_numpy(np.ascontiguousarray(self.ref.W[i])))
                self.b[i].copy_(torch.from_numpy(np.ascontiguousarray(self.ref.b[i])))


class TorchAdam:
    """fusion.nn.Adam on device tensors (coupled L2 on matrices only)."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = [torch.zeros_like(p) for p in params]
        self.v = [torch.zeros_like(p) for p in params]
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if self.wd and p.ndim > 1:
                g = g + self.wd * p
            m.mul_(self.b1).add_((1 - self.b1) * g)
            v.mul_(self.b2).add_((1 - self.b2) * g * g)
            p.sub_(self.lr * (m / c1) / (torch.sqrt(v / c2) + self.eps))

    def zero_grad(self):
        for p in self.params:
            p.grad = None


class DeviceLayout:
    """Record slicing and scales of fusion.schema.Layout, on a device."""

    def __init__(self, layout: Layout, dev):
        self.layout = layout
        self.blocks = layout.blocks
        self.scales = {g: torch.tensor(layout.scale(g).astype(np.float32), device=dev)
                       for g in ("global", "unit", "move", "cand_f")}
        self.vscale = torch.tensor(layout.value_scale(), dtype=torch.float64, device=dev)

    def get(self, rec, name):
        offset, shape = self.blocks[name]
        size = int(np.prod(shape))
        return rec[..., offset:offset + size].reshape(*rec.shape[:-1], *shape)


class DevicePolicyBatch:
    """fusion.models.PolicyBatch built on the device from raw int16 records."""

    def __init__(self, rec, dl: DeviceLayout):
        f = torch.float32
        g = dl.get(rec, "g").to(f) / dl.scales["global"]
        u = dl.get(rec, "u").to(f) / dl.scales["unit"]
        m = dl.get(rec, "m").to(f) / dl.scales["move"]
        self.cand_f = dl.get(rec, "cand_f").to(f) / dl.scales["cand_f"]
        B = rec.shape[0]
        self.mover_in = torch.cat([g[:, None, None, :].expand(B, 3, 5, g.shape[1]),
                                   u[:, :, None, :].expand(B, 3, 5, u.shape[2]), m], dim=-1)
        self.squad_in = torch.cat([g, u.reshape(B, -1), m.reshape(B, -1)], dim=-1)
        self.legal = dl.get(rec, "legal") != 0
        self.act_legal = dl.get(rec, "act_legal") != 0
        self.cand_valid = dl.get(rec, "cand") >= 0
        self._b = torch.arange(B, device=rec.device)[:, None]
        self._k = torch.arange(3, device=rec.device)[None, :]

    @property
    def joint_mask(self):
        l = self.legal
        return (l[:, 0, :, None, None] & l[:, 1, None, :, None] & l[:, 2, None, None, :]).reshape(-1, JOINT)

    def act_in(self, move):
        return self.mover_in[self._b, self._k, move]

    def act_mask(self, move):
        return self.act_legal[self._b, self._k, move]

    def target_in(self, move):
        cf = self.cand_f
        idx = move[:, :, None, None].expand(cf.shape[0], 3, cf.shape[2], 1)
        hit = torch.gather(cf[..., 4:9], -1, idx)[..., 0]
        return torch.stack([hit if c is None else cf[..., c] for c in TARGET_COLS], dim=-1)


def combine_joint(s, r):
    j = (s[:, 0, :, None, None] + s[:, 1, None, :, None]) + s[:, 2, None, None, :]
    return j.reshape(-1, JOINT) + r


class TorchPolicy:
    """Device training twin of a PolicyNet (uncalibrated logits)."""

    def __init__(self, model: PolicyNet, dev):
        self.model = model
        self.nets = {k: TorchMLP(n, dev) for k, n in model.nets.items()}

    @property
    def params(self):
        return [p for n in self.nets.values() for p in n.params]

    def loss(self, X: DevicePolicyBatch, y: dict, w, cfg: dict):
        """(loss tensor for backward, dict of detached per-head weighted means)."""
        wsum = w.sum().clamp(min=1e-12)
        s = self.nets["mover"].forward(X.mover_in)[..., 0]
        r = self.nets["squad"].forward(X.squad_in)
        lj = _SoftmaxXent.apply(combine_joint(s, r), X.joint_mask, y["y_joint"], w, float(cfg.get("smoothing", 0.0)))
        move = y["y_move"]
        za = self.nets["act"].forward(X.act_in(move))
        w3 = w.repeat_interleave(3)
        la = _SoftmaxXent.apply(za.reshape(-1, 4), X.act_mask(move).reshape(-1, 4), y["y_act"].reshape(-1), w3, 0.0)
        zt = self.nets["target"].forward(X.target_in(move))[..., 0]
        C = zt.shape[-1]
        lt = _SoftmaxXent.apply(zt.reshape(-1, C), X.cand_valid.reshape(-1, C), y["y_tgt"].reshape(-1), w3, 0.0)
        aw, tw = cfg.get("act_weight", 1.0), cfg.get("target_weight", 1.0)
        total = (lj + aw * la + tw * lt) / wsum
        return total, {"joint": lj.detach() / wsum, "act": la.detach() / wsum, "target": lt.detach() / wsum,
                       "total": total.detach()}

    def to_ref(self):
        for n in self.nets.values():
            n.to_ref()

    def from_ref(self):
        for n in self.nets.values():
            n.from_ref()


class TorchValue:
    """Device training twin of a ValueNet."""

    def __init__(self, model: ValueNet, dev):
        self.model = model
        self.net = TorchMLP(model.net, dev)

    @property
    def params(self):
        return self.net.params

    def loss(self, x, y, w, cfg: dict):
        wsum = w.sum().clamp(min=1e-12)
        aux = cfg.get("aux_weight", 0.25)
        head_w = torch.tensor([1.0, aux, aux], dtype=torch.float32, device=x.device)
        z = self.net.forward(x)
        total_sum = _BCE.apply(z, y, w[:, None] * head_w[None, :])
        outcome = _BCE.apply(z[:, 0].detach(), y[:, 0], w) / wsum
        return total_sum / wsum, {"outcome": outcome.detach(), "total": total_sum.detach() / wsum}

    def to_ref(self):
        self.net.to_ref()

    def from_ref(self):
        self.net.from_ref()


class RowSource:
    """Training rows on the device: resident when they fit the budget, else streamed in windows of whole
    minibatches gathered on the host (next window prepared in a background thread). Both paths visit rows in
    the order of `order` exactly."""

    def __init__(self, arrays: dict, dev, budget_bytes: float, window_rows: int):
        self.arrays, self.dev = arrays, dev
        self.n = len(next(iter(arrays.values())))
        size = sum(a.nbytes for a in arrays.values())
        self.resident = size <= budget_bytes
        self.window_rows = window_rows
        self.on_dev = {k: torch.from_numpy(np.ascontiguousarray(a)).to(dev) for k, a in arrays.items()} if self.resident else None

    def _gather(self, idx):
        return {k: np.ascontiguousarray(a[idx]) for k, a in self.arrays.items()}

    def _upload(self, host):
        pin = self.dev.type == "cuda"
        return {k: (torch.from_numpy(v).pin_memory() if pin else torch.from_numpy(v)).to(self.dev, non_blocking=pin)
                for k, v in host.items()}

    def minibatches(self, order: np.ndarray, batch: int):
        if self.resident:
            perm = torch.from_numpy(order).to(self.dev)
            for s in range(0, len(order), batch):
                sel = perm[s:s + batch]
                yield {k: v[sel] for k, v in self.on_dev.items()}
            return
        window = max(batch, (self.window_rows // batch) * batch)
        starts = list(range(0, len(order), window))
        result = {}

        def prepare(i):
            result[i] = self._gather(order[starts[i]:starts[i] + window])

        thread = threading.Thread(target=prepare, args=(0,))
        thread.start()
        for i in range(len(starts)):
            thread.join()
            host = result.pop(i)
            if i + 1 < len(starts):
                thread = threading.Thread(target=prepare, args=(i + 1,))
                thread.start()
            dev = self._upload(host)
            n = len(next(iter(dev.values())))
            for s in range(0, n, batch):
                yield {k: v[s:s + batch] for k, v in dev.items()}


LABELS = ("y_move", "y_joint", "y_act", "y_tgt")


class PolicyTrainer:
    """Drives TorchPolicy for training.common.fit: train_epoch(rng), val_loss(), state(), load(), opt.lr."""

    def __init__(self, model: PolicyNet, layout: Layout, train: dict, val: dict, cfg: dict, args, dev):
        self.model, self.cfg, self.batch, self.dev = model, cfg, args.batch, dev
        self.dl = DeviceLayout(layout, dev)
        self.net = TorchPolicy(model, dev)
        self.opt = TorchAdam(self.net.params, lr=args.lr, weight_decay=args.weight_decay)
        cols = ("rec", "weight") + LABELS
        budget = args.gpu_resident_gb * 2 ** 30
        self.train = RowSource({k: train[k] for k in cols}, dev, budget, args.gpu_window_rows)
        self.val = RowSource({k: val[k] for k in cols}, dev, budget, args.gpu_window_rows)
        self.val_wsum = float(val["weight"].sum())
        self.timing = {"train_seconds": 0.0, "val_seconds": 0.0, "rows_per_second": []}

    def _y(self, b):
        return {k: b[k].long() for k in LABELS}

    def train_epoch(self, rng):
        t0 = time.time()
        order = rng.permutation(self.train.n)
        total = torch.zeros((), dtype=torch.float64, device=self.dev)
        wsum = torch.zeros((), dtype=torch.float64, device=self.dev)
        for b in self.train.minibatches(order, self.batch):
            X = DevicePolicyBatch(b["rec"], self.dl)
            w = b["weight"]
            self.opt.zero_grad()
            loss, out = self.net.loss(X, self._y(b), w, self.cfg)
            loss.backward()
            self.opt.step()
            total += out["total"].double() * w.sum().double()
            wsum += w.sum().double()
        value = float(total / wsum)                         # one synchronization per epoch
        dt = time.time() - t0
        self.timing["train_seconds"] += dt
        self.timing["rows_per_second"].append(round(self.train.n / dt))
        return value

    @torch.no_grad()
    def evaluate(self, size=4096):
        t0 = time.time()
        total = torch.zeros((), dtype=torch.float64, device=self.dev)
        for b in self.val.minibatches(np.arange(self.val.n), size):
            X = DevicePolicyBatch(b["rec"], self.dl)
            w = b["weight"]
            _, out = self.net.loss(X, self._y(b), w, self.cfg)
            total += out["total"].double() * w.sum().double() / self.val_wsum
        self.timing["val_seconds"] += time.time() - t0
        return float(total)

    def state(self):
        self.net.to_ref()
        return self.model.state()

    def load(self, st):
        self.model.load(st)
        self.net.from_ref()


class ValueTrainer:
    """Drives TorchValue for training.common.fit; the 54 value columns are kept as int16 (resident if they fit)."""

    def __init__(self, model: ValueNet, layout: Layout, train: dict, val: dict, cfg: dict, args, dev):
        self.model, self.cfg, self.batch, self.dev = model, cfg, args.batch, dev
        self.dl = DeviceLayout(layout, dev)
        self.net = TorchValue(model, dev)
        self.opt = TorchAdam(self.net.params, lr=args.lr, weight_decay=args.weight_decay)
        budget = args.gpu_resident_gb * 2 ** 30
        slim = lambda d: {"v": np.ascontiguousarray(layout.get(d["rec"], "v")), "v_y": d["v_y"], "weight": d["weight"]}  # noqa: E731
        self.train = RowSource(slim(train), dev, budget, args.gpu_window_rows)
        self.val = RowSource(slim(val), dev, budget, args.gpu_window_rows)
        self.val_wsum = float(val["weight"].sum())
        self.timing = {"train_seconds": 0.0, "val_seconds": 0.0, "rows_per_second": []}

    def _x(self, v):
        return (v.double() / self.dl.vscale).float()          # fusion.models.value_inputs: float64 division, float32

    def train_epoch(self, rng):
        t0 = time.time()
        order = rng.permutation(self.train.n)
        total = torch.zeros((), dtype=torch.float64, device=self.dev)
        wsum = torch.zeros((), dtype=torch.float64, device=self.dev)
        for b in self.train.minibatches(order, self.batch):
            w = b["weight"]
            self.opt.zero_grad()
            loss, out = self.net.loss(self._x(b["v"]), b["v_y"], w, self.cfg)
            loss.backward()
            self.opt.step()
            total += out["outcome"].double() * w.sum().double()
            wsum += w.sum().double()
        value = float(total / wsum)
        dt = time.time() - t0
        self.timing["train_seconds"] += dt
        self.timing["rows_per_second"].append(round(self.train.n / dt))
        return value

    @torch.no_grad()
    def evaluate(self, size=8192):
        t0 = time.time()
        total = torch.zeros((), dtype=torch.float64, device=self.dev)
        for b in self.val.minibatches(np.arange(self.val.n), size):
            w = b["weight"]
            _, out = self.net.loss(self._x(b["v"]), b["v_y"], w, self.cfg)
            total += out["outcome"].double() * w.sum().double() / self.val_wsum
        self.timing["val_seconds"] += time.time() - t0
        return float(total)

    def state(self):
        self.net.to_ref()
        return self.model.state()

    def load(self, st):
        self.model.load(st)
        self.net.from_ref()


def peak_memory(dev) -> dict:
    if dev.type != "cuda":
        return {}
    return {"peak_allocated_gb": round(torch.cuda.max_memory_allocated(dev) / 2 ** 30, 3),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(dev) / 2 ** 30, 3)}
