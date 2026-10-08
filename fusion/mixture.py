"""Mixture weight, temperature calibration and KL-regularized fusion (docs/DESIGN.md sections 2 and 4).

Both fits minimize a convex function of one scalar, so bisection on the
derivative finds the global minimum to machine precision.

The deployment adapter (inference/adapter.hpp) decides which distribution the
bot actually samples, so the gate must fit and score exactly that one:

  joint-v2     one mixture choice per opponent team: with probability alpha the
               whole team move comes from the 125-way network joint, else each
               unit from the legacy forecast. q(m) = (1-a) prod_k qH_k(m_k) + a qNN(m).
  marginal-v1  each unit samples its own mixed marginal, independently:
               q(m) = prod_k [(1-a) qH_k(m_k) + a qNN_k(m_k)], qNN_k the network's marginal.

The two differ: v1 keeps none of the network's coordination (a network that is
only right about joint structure gains nothing under v1). Both share the same
per-unit marginals (1-a) qH_k + a qNN_k.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-300
ADAPTERS = {"marginal-v1": 1, "joint-v2": 2}  # values = inference/adapter.hpp Adapter


def _factors(q) -> np.ndarray:
    """(n,) one factor per row, or (n, K) a product of K mixture factors per row."""
    q = np.asarray(q, np.float64)
    return q[:, None] if q.ndim == 1 else q


def mixture_loss(alpha: float, qh: np.ndarray, qt: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted mean of -log((1 - alpha) qH + alpha qtheta) at the observed actions.

    qh, qt of shape (n,): one mixture per row. Shape (n, K): the row's probability
    is the product of K independent mixtures (the marginal-v1 adapter, K = 3 units).
    """
    qh, qt = _factors(qh), _factors(qt)
    w = np.ones(len(qh)) if w is None else np.asarray(w, np.float64)
    ll = np.log(np.maximum((1 - alpha) * qh + alpha * qt, EPS)).sum(1)
    return float(-(w * ll).sum() / w.sum())


def fit_alpha(qh: np.ndarray, qt: np.ndarray, w: np.ndarray | None = None, iters: int = 200) -> tuple[float, float]:
    """alpha* = argmin over [0, 1] of the mixture log-loss.

    qh, qt: probabilities each predictor gave to the action actually taken, shape
    (n,) or (n, K) as in mixture_loss. Each factor's -log is convex in alpha
    (second derivative (qt - qh)^2 / q_alpha^2 >= 0), so is their weighted sum;
    its derivative is nondecreasing and bisection on its sign is exact.
    """
    qh, qt = _factors(qh), _factors(qt)
    w = np.ones(len(qh)) if w is None else np.asarray(w, np.float64)

    def slope(a):
        return float(-(w[:, None] * (qt - qh) / np.maximum((1 - a) * qh + a * qt, EPS)).sum())

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


def adapter_prob(adapter: str, alpha: float, qh_units: np.ndarray, nn_joint: np.ndarray, nn_units: np.ndarray) -> np.ndarray:
    """Probability the deployed adapter gives the observed joint move, per row.

    qh_units, nn_units: (n, 3) legacy and network unit marginals at the observed
    unit moves; nn_joint: (n,) the network joint probability of the observed move.
    """
    if adapter == "joint-v2":
        return (1 - alpha) * np.prod(qh_units, 1) + alpha * nn_joint
    if adapter == "marginal-v1":
        return np.prod((1 - alpha) * qh_units + alpha * nn_units, 1)
    raise ValueError(f"unknown adapter {adapter!r} (known: {', '.join(ADAPTERS)})")


def adapter_inputs(adapter: str, qh_units: np.ndarray, nn_joint: np.ndarray, nn_units: np.ndarray):
    """(qh, qt) for fit_alpha / mixture_loss that score exactly `adapter`."""
    if adapter == "joint-v2":
        return np.prod(qh_units, 1), nn_joint
    if adapter == "marginal-v1":
        return qh_units, nn_units
    raise ValueError(f"unknown adapter {adapter!r} (known: {', '.join(ADAPTERS)})")


def adapter_table(adapter: str, alpha: float, qh: np.ndarray, nn: np.ndarray) -> np.ndarray:
    """Full 125-way distribution of one team move under `adapter`; qh (3, 5) unit marginals, nn (125,) joint."""
    nn = np.asarray(nn, np.float64).reshape(5, 5, 5)
    qh = np.asarray(qh, np.float64)
    if adapter == "joint-v2":
        legacy = qh[0][:, None, None] * qh[1][None, :, None] * qh[2][None, None, :]
        return ((1 - alpha) * legacy + alpha * nn).reshape(-1)
    marg = [nn.sum((1, 2)), nn.sum((0, 2)), nn.sum((0, 1))]
    mix = [(1 - alpha) * qh[k] + alpha * marg[k] for k in range(3)]
    if adapter == "marginal-v1":
        return (mix[0][:, None, None] * mix[1][None, :, None] * mix[2][None, None, :]).reshape(-1)
    raise ValueError(f"unknown adapter {adapter!r} (known: {', '.join(ADAPTERS)})")


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
