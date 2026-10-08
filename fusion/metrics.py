"""Evaluation metrics. Every function takes optional importance weights."""
from __future__ import annotations

import numpy as np


def _w(w, n):
    return np.ones(n) if w is None else np.asarray(w, np.float64)


def log_loss(p_true: np.ndarray, w=None) -> float:
    """Weighted mean negative log-probability of the observed outcome (nats)."""
    w = _w(w, len(p_true))
    return float(-(w * np.log(np.maximum(p_true, 1e-300))).sum() / w.sum())


def rate(hit: np.ndarray, w=None) -> float:
    """Weighted share of true entries (accuracy when hit = prediction == label)."""
    w = _w(w, len(hit))
    return float((w * hit).sum() / w.sum())


def ece(confidence: np.ndarray, correct: np.ndarray, w=None, bins: int = 10) -> float:
    """Expected calibration error of top-1 confidence."""
    w = _w(w, len(correct))
    idx = np.minimum((confidence * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ws = w[sel].sum()
            total += ws * abs((w[sel] * correct[sel]).sum() / ws - (w[sel] * confidence[sel]).sum() / ws)
    return float(total / w.sum())


def binned_calibration(p: np.ndarray, y: np.ndarray, w=None, bins: int = 10) -> float:
    """Weighted mean over probability bins of |mean target - mean prediction| (fractional targets allowed)."""
    w = _w(w, len(y))
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ws = w[sel].sum()
            total += ws * abs((w[sel] * (y[sel] - p[sel])).sum() / ws)
    return float(total / w.sum())


def brier(p: np.ndarray, y: np.ndarray, w=None) -> float:
    w = _w(w, len(y))
    return float((w * (p - y) ** 2).sum() / w.sum())


def bce(p: np.ndarray, y: np.ndarray, w=None) -> float:
    w = _w(w, len(y))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())


def pair_ranking_accuracy(pred: np.ndarray, target: np.ndarray, group: np.ndarray) -> tuple[float, int]:
    """Within each group (one match turn), the share of team pairs with different
    targets that the prediction orders the same way. Returns (accuracy, pairs)."""
    order = np.argsort(group, kind="stable")
    g, p, t = group[order], pred[order], target[order]
    cuts = np.flatnonzero(np.diff(g)) + 1
    right = total = 0
    for gp, gt in zip(np.split(p, cuts), np.split(t, cuts)):
        dt = np.sign(gt[:, None] - gt[None, :])
        dp = np.sign(gp[:, None] - gp[None, :])
        upper = np.triu(dt != 0, 1)
        total += int(upper.sum())
        right += int((upper & (dt == dp)).sum())
    return (right / total if total else float("nan")), total


def clustered_mean(values: np.ndarray, w, clusters: np.ndarray) -> tuple[float, float, int]:
    """Weighted mean of per-row values with a standard error clustered by `clusters` (e.g. map seed).

    Rows of one game or map share its layout and trajectory, so they are not
    independent; the SE treats each cluster as one observation. Returns
    (mean, se, number of clusters); se is inf with fewer than two clusters.
    """
    values = np.asarray(values, np.float64)
    w = _w(w, len(values))
    mean = float((w * values).sum() / w.sum())
    _, inv = np.unique(clusters, return_inverse=True)
    k = int(inv.max()) + 1 if len(inv) else 0
    if k < 2:
        return mean, float("inf"), k
    s = np.bincount(inv, weights=w * (values - mean), minlength=k)
    se = float(np.sqrt((s ** 2).sum() * k / (k - 1)) / w.sum())
    return mean, se, k
