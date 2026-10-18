"""Mechanics parity: agent_v2's native simulator step against the official engine (task T0085).

Both implementations get the same full commands: random orders for every unit (moves, waits, pickups, drops,
attacks on enemies within range 3 or on enemies that respawn this step; a flag carrier usually steps toward its own
base, so captures happen), from a seeded generator. The engine steps them; the native step
(tools/cpp/agent_v2/check.cpp `check step`, the agent_v2 simulator's fusion::step) gets the same block and orders,
and the engine's flag refresh outcomes as controlled spots. The successors are compared field by field: turn,
scores, every unit's position, HP, carried flag and respawn clock, every flag's status, position, holder and return
clock, and the attack list.

Scope, stated in every report:
  * refreshes are controlled: the native step is given the engine's refresh positions, so this checks the refresh
    mechanics, not that the native sampler draws the engine's distribution;
  * respawn immunity is not compared as a flag (`check step` reports respawn clocks, not the immune flag); its
    effect is compared through the attack list and HP, and coverage requires attacks on and by units that are
    immune during the step;
  * no P is involved: this separates mechanics from opponent-model error.

Coverage counts the engine's own events of the compared steps (death, respawn, pickup, drop, capture, return,
attack), targets hit by two or more attackers, flag refreshes, and attack orders on or by a unit that is immune
during the step (it respawned at the step's start). A report passes only with zero
differences AND every coverage minimum met; otherwise it is "failed" (differences) or "insufficient_coverage".
Zero compared steps is never a pass.
"""
from __future__ import annotations

import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from fusion import runner
from fusion.paths import REPO
from tools.search_diagnostics.schemas import SCHEMA_PARITY, ValidationError, sha256_file

FLAG_STATUS = {"home": 0, "carried": 1, "dropped": 2, "cooldown": 3}
COVERAGE = ("death", "respawn", "pickup", "drop", "capture", "return", "attack", "simultaneous_attack", "refresh",
            "attack_on_immune", "attack_by_immune")
SCOPE = ("refreshes controlled (engine refresh positions given to the native step); respawn immunity compared only "
         "through attacks and HP; no P involved")


def parse_engine_block(text: str) -> dict[str, Any]:
    st: dict[str, Any] = {"units": [], "flags": [], "attacks": []}
    for line in text.strip().split("\n"):
        w = line.split()
        if not w:
            continue
        if w[0] == "TURN":
            st["turn"] = int(w[1])
        elif w[0] == "SCORE":
            st["score"] = list(map(int, w[1:]))
        elif w[0] == "U":
            st["units"].append(tuple(map(int, w[3:8])))
        elif w[0] == "F":
            st["flags"].append((FLAG_STATUS[w[2]], *map(int, w[3:7])))
        elif w[0] == "E" and len(w) >= 4 and w[1] == "attack":
            st["attacks"].append((int(w[2]), int(w[3])))
    st["attacks"].sort()
    return st


def parse_step_output(line: str, teams: int, n_units: int, n_flags: int) -> dict[str, Any]:
    head, sep, att = line.partition(" |")
    if not sep:
        raise ValidationError(f"check step printed {line[:80]!r}")
    t = list(map(int, head.split()))
    if len(t) != 1 + teams + 5 * n_units + 5 * n_flags:
        raise ValidationError(f"check step printed {len(t)} numbers")
    k = 1 + teams
    units = [tuple(t[k + 5 * i:k + 5 * i + 5]) for i in range(n_units)]
    k += 5 * n_units
    flags = [tuple(t[k + 5 * i:k + 5 * i + 5]) for i in range(n_flags)]
    a = list(map(int, att.split()))
    return {"turn": t[0], "score": t[1:1 + teams], "units": units, "flags": flags, "attacks": sorted(zip(a[::2], a[1::2]))}


def build_check(out: Path) -> Path:
    """tools/cpp/agent_v2/check.cpp with the flags tests/test_agent_v2.py uses (the step mode needs no model)."""
    actors = REPO / "inference" / "meow_rl" / "aligned" / "p2_large"
    return runner.compile_cpp(REPO / "tools" / "cpp" / "agent_v2" / "check.cpp", out, [
        f"-I{REPO / 'inference'}", f"-I{REPO / 'tools' / 'cpp' / 'meow_rl'}", "-DFUSION_TIER_PXXL_VM=1",
        f'-DACTOR_HEADER="{(actors / "actor.hpp").as_posix()}"',
        f'-DCRITIC_HEADER="{(actors / "critic_fp32.hpp").as_posix()}"'])


DELTAS = ((0, 0), (0, -1), (0, 1), (-1, 0), (1, 0))   # S U D L R


def bases(init: str) -> list[tuple[int, int]]:
    lines = init.split("\n")
    i = next(k for k, ln in enumerate(lines) if ln.startswith("BASES "))
    return [tuple(map(int, lines[i + 1 + t].split())) for t in range(int(lines[i].split()[1]))]


def random_orders(st: dict, rnd: random.Random, home: list[tuple[int, int]] | None = None) -> list[tuple[int, int, int]]:
    orders = []
    respawning = [j for j, u in enumerate(st["units"]) if u[0] < 0 and 0 <= u[4] <= st["turn"]]
    for uid, (x, y, hp, flag, _) in enumerate(st["units"]):
        foes = [j for j, u in enumerate(st["units"])
                if j // 3 != uid // 3 and u[0] >= 0 and abs(u[0] - x) + abs(u[1] - y) <= 3]
        if respawning and rnd.random() < 0.15:
            foes += [j for j in respawning if j // 3 != uid // 3]
        r = rnd.random()
        if flag >= 0 and home and r >= 0.05:
            bx, by = home[uid // 3]
            best = min(range(5), key=lambda m: (abs(x + DELTAS[m][0] - bx) + abs(y + DELTAS[m][1] - by), rnd.random()))
            orders.append((best if rnd.random() < 0.8 else rnd.randrange(5), 0, -1))
        elif flag >= 0 and r < 0.05:
            orders.append((rnd.randrange(5), 2, -1))
        elif r < 0.55 and foes:
            orders.append((rnd.randrange(5), 3, rnd.choice(foes)))
        elif r < 0.8:
            orders.append((rnd.randrange(5), 1, -1))
        else:
            orders.append((rnd.randrange(5), 0, -1))
    return orders


def audit(engine, check_exe: Path, counts: list[int], steps: int, seed: int,
          minimums: dict[str, int] | None = None) -> dict:
    if not counts or steps < 1:
        raise ValidationError("parity needs team counts and at least one step per count")
    minimums = {k: 1 for k in COVERAGE} | (minimums or {})
    rnd = random.Random(seed)
    diffs, covered, compared = [], Counter(), 0
    for teams in counts:
        g = engine.Game(teams, rnd.randrange(1, 2 ** 31), flag_seed=rnd.randrange(1, 2 ** 62))
        first = g.encode(0)
        init, block = first[:first.index("TURN")], first[first.index("TURN"):]
        home = bases(init)
        for _ in range(steps):
            if g.done:
                break
            st = parse_engine_block(block)
            orders = random_orders(st, rnd, home)
            cmds = {}
            for t in range(teams):
                toks = [str(g.turn + 1)]
                for m, a, tg in orders[3 * t:3 * t + 3]:
                    toks += ["SUDLR"[m], "X" if a == 2 else "-" if a == 0 else "P" if a == 1 else str(tg)]
                cmds[t] = g.decode(t, " ".join(toks))
            g.step(cmds)
            nxt = g.encode(0)
            want = parse_engine_block(nxt)
            events = Counter(e["type"] for e in g.events)
            for kind in ("death", "respawn", "pickup", "drop", "capture", "return", "attack"):
                covered[kind] += events[kind]
            immune = {u["id"] for u in g.units if u["immune"]}
            covered["attack_on_immune"] += sum(1 for _, a, tg in orders if a == 3 and tg in immune)
            covered["attack_by_immune"] += sum(1 for i, (_, a, _) in enumerate(orders) if a == 3 and i in immune)
            hits = Counter(e["target"] for e in g.events if e["type"] == "attack")
            covered["simultaneous_attack"] += sum(1 for v in hits.values() if v >= 2)
            # Refreshes are the engine's return events; a flag picked up in the step it returned sits at its
            # carrier's cell (the carrier stood on the spot), so that cell is the controlled spot.
            refreshed = {e["flag"] for e in g.events if e["type"] == "return"}
            covered["refresh"] += len(refreshed)

            def spot(f):
                status, x, y, holder, _ = want["flags"][f]
                return (want["units"][holder][0], want["units"][holder][1]) if status == 1 else (x, y)
            spots = [spot(f) if f in refreshed else (-1, -1) for f in range(len(st["flags"]))]
            text = (init + block + f"{rnd.randrange(1, 2 ** 62)}\n" + "".join(f"{m} {a} {t}\n" for m, a, t in orders)
                    + "spots\n" + "".join(f"{x} {y}\n" for x, y in spots))
            r = subprocess.run([str(check_exe), "step"], input=text, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                raise ValidationError(f"check step failed: {(r.stdout or r.stderr)[:200]}")
            got = parse_step_output(r.stdout.strip(), teams, len(st["units"]), len(st["flags"]))
            compared += 1
            for k in ("turn", "score", "units", "flags", "attacks"):
                if got[k] != want[k]:
                    diffs.append({"teams": teams, "turn": want["turn"], "field": k, "engine": want[k], "native": got[k]})
            block = nxt
    short = {k: (covered[k], n) for k, n in minimums.items() if covered[k] < n}
    status = "failed" if diffs else "insufficient_coverage" if short or not compared else "passed"
    return {
        "schema": SCHEMA_PARITY, "status": status, "steps_compared": compared, "team_counts": counts,
        "steps_per_count": steps, "seed": seed, "check_binary_sha256": sha256_file(check_exe),
        "engine_sha256": sha256_file(engine.__file__),
        "coverage": {k: covered[k] for k in COVERAGE}, "minimums": minimums,
        "short_of_minimum": {k: {"seen": s, "needed": n} for k, (s, n) in short.items()},
        "differences": len(diffs), "first_differences": diffs[:10], "scope": SCOPE,
    }
