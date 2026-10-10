"""Synthetic seat streams for correctness checks that need no AI_Test checkout.

The stream is protocol-valid (tools/cpp/stream.hpp parses it): an open map of the
team count's size, bases on a ring, three units per team that step left and
right every turn (so value features change between turns), and scores that
change now and then. It is not a game: timing benchmarks must replay recorded
matches instead (tools/host/capacity_bench.py).
"""
from __future__ import annotations

import math

from .paths import load_profile


def stream(teams: int, turns: int = 40, size: int | None = None) -> str:
    size = size or load_profile(teams)["map_size"]
    c, r = (size - 1) / 2, (size - 1) / 2 - 4
    bases = [(round(c + r * math.cos(2 * math.pi * t / teams)), round(c + r * math.sin(2 * math.pi * t / teams)))
             for t in range(teams)]
    flags = max(2, math.ceil(teams / 2))
    spots = [(round(c + (r / 2) * math.cos(2 * math.pi * f / flags + 0.5)), round(c + (r / 2) * math.sin(2 * math.pi * f / flags + 0.5)))
             for f in range(flags)]
    lines = [f"INIT {teams} 0 {size} {turns} 100 34 10 15 8", "MAP", *["." * size] * size, f"BASES {teams}",
             *[f"{x} {y}" for x, y in bases], f"SPOTS {flags}", *[f"{x} {y}" for x, y in spots]]
    for t in range(1, turns + 1):
        score = [(t // 7 + 3 * team) % 5 for team in range(teams)]
        lines += [f"TURN {t}", "SCORE " + " ".join(map(str, score)), f"UNITS {3 * teams}"]
        for uid in range(3 * teams):
            bx, by = bases[uid // 3]
            lines.append(f"U {uid} {uid // 3} {bx + uid % 3 - 1 + (t % 2)} {by} 100 -1 -1")
        lines += [f"FLAGS {flags}", *[f"F {f} home {x} {y} -1 -1" for f, (x, y) in enumerate(spots)], "EVENTS 0", "END"]
    return "\n".join(lines) + "\n"
