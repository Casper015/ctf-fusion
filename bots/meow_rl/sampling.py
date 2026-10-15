"""Python mirror of bots/meow_rl/sample.hpp (the bot's sampled actor-direct choice) and of its greedy mode.

Used by tests/test_meow_rl_export.py to replay recorded streams through training.meow_rl's bridge and compare every
reply with the compiled bot.
"""
from __future__ import annotations

import math

import numpy as np

from training.meow_rl import quant as Q
from training.meow_rl.policy import JA, JB, JC

MASK64 = (1 << 64) - 1
SALT = "meow_rl_actor_v1\n"
GOLDEN = 0x9E3779B97F4A7C15


def fnv1a(text: str) -> int:
    h = 1469598103934665603
    for c in text.encode():
        h = ((h ^ c) * 1099511628211) & MASK64
    return h


def parse_init(block: str) -> dict:
    """teams, seat, size, turns, map rows, bases, spots of the first (INIT) block."""
    tok = block.split("\n")
    head = tok[0].split()
    assert head[0] == "INIT"
    teams, me, size, turns = (int(x) for x in head[1:5])
    i = tok.index("MAP") + 1
    rows = tok[i:i + size]
    i = next(k for k, line in enumerate(tok) if line.startswith("BASES"))
    bases = [tuple(int(v) for v in tok[i + 1 + k].split()) for k in range(int(tok[i].split()[1]))]
    i = next(k for k, line in enumerate(tok) if line.startswith("SPOTS"))
    spots = [tuple(int(v) for v in tok[i + 1 + k].split()) for k in range(int(tok[i].split()[1]))]
    return {"teams": teams, "me": me, "size": size, "turns": turns, "rows": rows, "bases": bases, "spots": spots}


def seed(block: str) -> int:
    g = parse_init(block)
    s = SALT + f"{g['teams']} {g['me']} {g['size']} {g['turns']}\n"
    s += "".join(r + "\n" for r in g["rows"])
    s += "".join(f"{x} {y}\n" for x, y in g["bases"])
    s += "".join(f"{x} {y}\n" for x, y in g["spots"])
    return fnv1a(s)


def splitmix64(x: int) -> int:
    x = (x + GOLDEN) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def uniform(seed_: int, turn: int) -> float:
    return (splitmix64((seed_ + turn * GOLDEN) & MASK64) >> 11) * 2.0 ** -53


def draw(q: dict, feats: np.ndarray, mask: np.ndarray, u: float) -> int:
    s = Q.forward_int(q, feats)                                  # [3, 5] output accumulators
    z = np.ldexp((s[0, JA] + s[1, JB] + s[2, JC]).astype(np.float64), -q["net"]["out_exp"]) / q["temperature"]
    top = max(float(z[j]) for j in range(125) if mask[j])
    # math.exp is the C library's exp (as std::exp); sums run left to right over all 125 entries, as in sample.hpp
    p = [math.exp(float(z[j]) - top) if mask[j] else 0.0 for j in range(125)]
    total = 0.0
    for x in p:
        total += x
    target, acc, last = u * total, 0.0, 0
    for j in range(125):
        if mask[j]:
            acc += p[j]
            last = j
            if acc > target:
                return j
    return last


def choose(q: dict, feats: np.ndarray, mask: np.ndarray, seed_: int, turn: int, greedy: bool = False) -> int:
    return Q.greedy(q, feats, mask) if greedy else draw(q, feats, mask, uniform(seed_, turn))
