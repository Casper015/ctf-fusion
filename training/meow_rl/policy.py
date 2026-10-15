"""The masked joint actor: NumPy reference used by CPU rollout workers (no PyTorch needed there).

Per own unit k and move m the shared MLP 112 -> 64 -> 32 -> 1 (ReLU) scores the reference's move features x[k, m].
A joint movement j = 25 a + 5 b + c scores

    z_j = (f(x[0, a]) + f(x[1, b]) + f(x[2, c])) / T,        T = 1.05 (Planner::policy's temperature),

and the policy is the softmax of z over the feasible joints only (the bridge's joint mask):

    log pi(j | o) = z_j - logsumexp_{h feasible} z_h,         pi(j) = 0 for infeasible j.

With the reference weights this is the reference actor's product of per-unit move probabilities restricted to
feasible joints and renormalized (the warm start). The reference's search additionally pruned moves below
probability 0.00005: that pruning depends on the policy, so it is not part of the mask (identity.ACTOR_ARCH).
The sampled index, its log probability and the mask are what the trainer stores; nothing re-samples or repairs.
"""
from __future__ import annotations

import numpy as np

from .identity import JOINT, PF, TEMPERATURE

JA = np.arange(JOINT) // 25
JB = np.arange(JOINT) // 5 % 5
JC = np.arange(JOINT) % 5
ACTOR_SHAPES = {"w0": (64, PF), "b0": (64,), "w1": (32, 64), "b1": (32,), "w2": (1, 32), "b2": (1,)}


def check_actor(w: dict) -> dict:
    out = {}
    for key, shape in ACTOR_SHAPES.items():
        a = np.asarray(w[key], np.float32)
        if a.shape != shape:
            raise ValueError(f"actor {key}: shape {a.shape}, expected {shape}")
        if not np.isfinite(a).all():
            raise ValueError(f"actor {key}: non-finite weights")
        out[key] = a
    return out


def unit_scores(w: dict, feats: np.ndarray) -> np.ndarray:
    """f(x) for feats [..., 3, 5, 112] -> [..., 3, 5] (float32, the reference's ReLU MLP)."""
    x = np.asarray(feats, np.float32)
    h = np.maximum(0.0, x @ w["w0"].T + w["b0"])
    z = np.maximum(0.0, h @ w["w1"].T + w["b1"])
    return (z @ w["w2"].T + w["b2"])[..., 0]


def joint_logits(scores: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """z_j for scores [..., 3, 5] and mask [..., 125]; infeasible joints are -inf."""
    s = np.asarray(scores, np.float32) / np.float32(TEMPERATURE)
    z = s[..., 0, JA] + s[..., 1, JB] + s[..., 2, JC]
    return np.where(mask, z, -np.inf).astype(np.float32)


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = np.max(z, axis=-1, keepdims=True)
    if not np.isfinite(m).all():
        raise ValueError("no feasible joint move")
    e = np.exp(z - m)
    return (z - m - np.log(np.sum(e, axis=-1, keepdims=True))).astype(np.float32)


def joint_log_probs(w: dict, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return log_softmax(joint_logits(unit_scores(w, feats), mask))


def sample(logp: np.ndarray, u: float) -> int:
    """Inverse-CDF draw from exp(logp) with one uniform u in [0, 1); infeasible entries (-inf) are never chosen."""
    p = np.exp(logp.astype(np.float64))
    c = np.cumsum(p)
    j = int(np.searchsorted(c, u * c[-1], side="right"))
    j = min(j, JOINT - 1)
    while not np.isfinite(logp[j]):          # u * total landed exactly on a boundary before a masked tail
        j -= 1
    return j


def greedy(logp: np.ndarray) -> int:
    """The most probable feasible joint; ties go to the lowest index."""
    return int(np.argmax(logp))


def entropy(logp: np.ndarray) -> float:
    ok = np.isfinite(logp)
    p = np.exp(logp[ok].astype(np.float64))
    return float(-(p * logp[ok]).sum())


def unit_probs(w: dict, feats: np.ndarray, unit_legal: np.ndarray) -> np.ndarray:
    """Per-unit move probabilities as Planner::policy computes them (for the warm-start parity test)."""
    s = unit_scores(w, feats) / np.float32(TEMPERATURE)
    s = np.where(unit_legal, s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=-1, keepdims=True)
