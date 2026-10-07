"""Mixture weight, temperature calibration and KL-regularized fusion (docs/DESIGN.md sections 2 and 4).

Both fits minimize a convex function of one scalar, so bisection on the
derivative finds the global minimum to machine precision.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-300


def mixture_loss(alpha: float, qh: np.ndarray, qt: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted mean of -log((1 - alpha) qH + alpha qtheta) at the observed actions."""
    w = np.ones_like(qh) if w is None else w
    return float(-(w * np.log(np.maximum((1 - alpha) * qh + alpha * qt, EPS))).sum() / w.sum())


def fit_alpha(qh: np.ndarray, qt: np.ndarray, w: np.ndarray | None = None, iters: int = 200) -> tuple[float, float]:
    """alpha* = argmin over [0, 1] of the mixture log-loss.

    qh, qt: probabilities each predictor gave to the action actually taken.
    The loss is convex in alpha (second derivative sum w (qt - qh)^2 / q_alpha^2 >= 0),
    so its derivative is nondecreasing and bisection on its sign is exact.
    """
    qh, qt = np.asarray(qh, np.float64), np.asarray(qt, np.float64)
    w = np.ones_like(qh) if w is None else np.asarray(w, np.float64)

    def slope(a):
        return float(-(w * (qt - qh) / np.maximum((1 - a) * qh + a * qt, EPS)).sum())

    if slope(0.0) >= 0:
        a = 0.0
    elif slope(1.0) <= 0:
        a = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if slope(mid) < 0 else (lo, mid)
        a = 0.5 * (lo + hi)
    return a, mixture_loss(a, qh, qt, w)


def fit_temperature(logits: np.ndarray, mask: np.ndarray, label: np.ndarray, w: np.ndarray | None = None,
                    lo: float = 0.05, hi: float = 20.0, iters: int = 100) -> float:
    """Inverse temperature beta minimizing weighted NLL of softmax(beta * logits) over the mask.

    NLL(beta) = sum w (logsumexp(beta l) - beta l_y) is convex in beta, with
    derivative sum w (E_beta[l] - l_y).
    """
    ok = label >= 0
    l = np.where(mask, logits, 0.0)[ok].astype(np.float64)
    m = mask[ok]
    y = l[np.arange(len(l)), label[ok]]
    w = np.ones(len(l)) if w is None else np.asarray(w, np.float64)[ok]

    def slope(beta):
        z = np.where(m, beta * l, -np.inf)
        z = z - z.max(1, keepdims=True)
        p = np.where(m, np.exp(z), 0.0)
        p /= p.sum(1, keepdims=True)
        return float((w * ((p * l).sum(1) - y)).sum())

    if slope(lo) >= 0:
        return lo
    if slope(hi) <= 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slope(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def kl_fusion(log_p0: np.ndarray, q_value: np.ndarray, tau: float) -> np.ndarray:
    """mu*(a) proportional to p0(a) exp(Q(a) / tau): argmax of E_mu[Q] - tau KL(mu || p0).

    Entries with p0 = 0 (log_p0 = -inf) stay at zero, as the KL direction requires.
    """
    z = log_p0 + q_value / tau
    finite = np.isfinite(z)
    mx = np.max(np.where(finite, z, -np.inf), axis=-1, keepdims=True)
    e = np.where(finite, np.exp(np.where(finite, z - mx, 0.0)), 0.0)
    return e / e.sum(-1, keepdims=True)
