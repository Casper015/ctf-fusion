"""Numpy multilayer perceptrons with clipped ReLU, and Adam.

The models are small (tens of thousands of weights), so plain numpy trains
them at millions of samples per minute on one CPU and keeps training,
quantization and the C++ emulation in one language with no heavy dependency.
Hidden activations are clipped ReLU on [0, 1], the range inference/mlp.hpp
represents with 128 integer levels.
"""
from __future__ import annotations

import numpy as np


class MLP:
    """Dense layers; clipped ReLU between them, linear output. Works on any leading batch shape."""

    def __init__(self, sizes: list[int], rng: np.random.Generator, dtype=np.float32):
        self.sizes = list(sizes)
        self.W, self.b = [], []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            last = i == len(sizes) - 2
            limit = np.sqrt((1.0 if last else 3.0) / fan_in)
            self.W.append(rng.uniform(-limit, limit, (fan_out, fan_in)).astype(dtype))
            # Hidden biases start inside the clipped ReLU's linear range.
            self.b.append(np.full(fan_out, 0.0 if last else 0.1, dtype))
        self.gW = [np.zeros_like(w) for w in self.W]
        self.gb = [np.zeros_like(b) for b in self.b]
        self._cache = None

    @property
    def params(self):
        return self.W + self.b

    @property
    def grads(self):
        return self.gW + self.gb

    def zero_grad(self):
        for g in self.grads:
            g.fill(0)

    # numpy's Accelerate BLAS on macOS can raise spurious floating-point flags
    # inside matmul; training checks that every loss is finite instead.
    @np.errstate(divide="ignore", over="ignore", invalid="ignore")
    def forward(self, x: np.ndarray, keep: bool = True) -> np.ndarray:
        lead = x.shape[:-1]
        h = x.reshape(-1, x.shape[-1])
        acts, pres = [h], []
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = h @ w.T + b
            if i < len(self.W) - 1:
                pres.append(z)
                h = np.clip(z, 0.0, 1.0)
                acts.append(h)
            else:
                h = z
        if keep:
            self._cache = (lead, acts, pres)
        return h.reshape(*lead, h.shape[-1])

    @np.errstate(divide="ignore", over="ignore", invalid="ignore")
    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Accumulates parameter gradients; returns the gradient for the input."""
        lead, acts, pres = self._cache
        d = dy.reshape(-1, dy.shape[-1])
        for i in range(len(self.W) - 1, -1, -1):
            self.gW[i] += d.T @ acts[i]
            self.gb[i] += d.sum(0)
            d = d @ self.W[i]
            if i > 0:
                z = pres[i - 1]
                d = d * ((z > 0.0) & (z < 1.0))
        return d.reshape(*lead, d.shape[-1])

    def state(self) -> dict:
        return {f"W{i}": w for i, w in enumerate(self.W)} | {f"b{i}": b for i, b in enumerate(self.b)}

    def load(self, state: dict):
        for i in range(len(self.W)):
            self.W[i][...] = state[f"W{i}"]
            self.b[i][...] = state[f"b{i}"]


class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, beta1, beta2, eps, weight_decay
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            if self.wd and p.ndim > 1:
                g = g + self.wd * p
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(v / c2) + self.eps)


def masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log softmax over the entries where mask is true; -inf elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(mask, logits, -np.inf)
        mx = np.max(z, axis=-1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        lse = np.log(np.sum(np.where(mask, np.exp(z - mx), 0.0), axis=-1, keepdims=True)) + mx
        return np.where(mask, z - lse, -np.inf)


def softmax_xent(logits, mask, label, weight, smoothing=0.0):
    """Weighted cross-entropy with label smoothing over the legal support.

    Returns (sum of weighted losses, gradient w.r.t. logits). Rows with
    label < 0 or weight 0 contribute nothing.
    """
    logp = masked_log_softmax(logits, mask)
    p = np.where(mask, np.exp(logp), 0.0)
    ok = label >= 0
    target = np.zeros_like(p)
    idx = np.nonzero(ok)[0]
    target[idx, label[idx]] = 1.0
    if smoothing:
        legal = mask.sum(-1, keepdims=True).clip(min=1)
        target = np.where(mask, (1 - smoothing) * target + smoothing / legal, 0.0)
    w = np.where(ok, weight, 0.0)[:, None]
    loss = -(w * np.where(mask, target * np.where(mask, logp, 0.0), 0.0)).sum()
    grad = w * (p - target)
    return float(loss), grad


def bce_with_logits(z, y, weight):
    """Weighted binary cross-entropy (targets may be fractional). Returns (loss sum, grad)."""
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    grad = 1.0 / (1.0 + np.exp(-z)) - y
    return float((weight * loss).sum()), weight * grad
