"""Reward units and generalized advantage estimation (docs/MEOWMIND_RL_PLAN.md section 3).

Reward is 0 before termination and 2 U_i - 1 at termination, with the pairwise utility

    U_i = sum_{j != i} [1(z_i > z_j) + 0.5 * 1(z_i == z_j)] / (N - 1)

of the final score vector z (roster.strength.utility). gamma = 1, lambda = 0.95:

    delta_t = r_t + gamma * (1 - terminal_t) * V(o_{t+1}) - V(o_t)
    A_t     = delta_t + gamma * lambda * (1 - terminal_t) * A_{t+1}          (within one episode segment)
    R_t     = A_t + V(o_t)

A segment ends either in a true terminal (bootstrap 0) or in a truncation (bootstrap V of the observation that
followed the last stored decision). Segments never bootstrap across episodes.
"""
from __future__ import annotations

import numpy as np


def utility(scores, seat: int) -> float:
    own = scores[seat]
    return sum(1.0 if own > s else 0.5 if own == s else 0.0 for i, s in enumerate(scores) if i != seat) / (len(scores) - 1)


def terminal_reward(scores, seat: int) -> float:
    return 2.0 * utility(scores, seat) - 1.0


def gae(rewards: np.ndarray, values: np.ndarray, terminated: bool, bootstrap: float, gamma: float = 1.0,
        lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Advantages and returns of one episode segment (float64 accumulation, float32 results).

    rewards, values: [T]; terminated: the segment ends in a true terminal (then `bootstrap` must be 0); bootstrap:
    V of the observation after the last decision of a truncated segment."""
    r = np.asarray(rewards, np.float64)
    v = np.asarray(values, np.float64)
    if r.shape != v.shape or r.ndim != 1:
        raise ValueError("rewards and values must be equal-length vectors")
    if terminated and bootstrap != 0.0:
        raise ValueError("a true terminal bootstraps 0")
    if not np.isfinite(r).all() or not np.isfinite(v).all() or not np.isfinite(bootstrap):
        raise ValueError("non-finite reward or value")
    T = len(r)
    adv = np.zeros(T)
    nxt_v, nxt_a = float(bootstrap), 0.0
    for t in range(T - 1, -1, -1):
        delta = r[t] + gamma * nxt_v - v[t]
        nxt_a = delta + gamma * lam * nxt_a
        adv[t] = nxt_a
        nxt_v = v[t]
    return adv.astype(np.float32), (adv + v).astype(np.float32)
