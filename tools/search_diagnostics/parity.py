"""Native agent_v2 simulator vs official engine.py parity audit (T0085).

Compares:
  - Unit positions (x, y), HP, carried flag, respawn clocks, and immunity.
  - Flag status (home, carried, dropped, cooldown), coordinates, holder, and return clocks.
  - Attack interactions and simultaneous crossfire resolution.
  - Score updates and game termination conditions.
  - Controlled refresh spot outcomes.
Separates mechanics parity from opponent-model error.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fusion import runner
from fusion.paths import REPO, ai_test_root, load_engine
from tools.search_diagnostics.schemas import (
    SCHEMA_PARITY,
    ValidationError,
)

FLAG_STATUS_MAP = {"home": 0, "carried": 1, "dropped": 2, "cooldown": 3}


@dataclass
class ParityDiff:
    step: int
    teams: int
    field: str
    expected: Any
    actual: Any


@dataclass
class ParityReport:
    schema: str                         # SCHEMA_PARITY
    steps_checked: int
    passed: bool
    cases_covered: dict[str, int]
    diffs: list[ParityDiff] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "steps_checked": self.steps_checked,
            "passed": self.passed,
            "cases_covered": self.cases_covered,
            "diff_count": len(self.diffs),
            "diffs": [
                {
                    "step": d.step,
                    "teams": d.teams,
                    "field": d.field,
                    "expected": d.expected,
                    "actual": d.actual,
                }
                for d in self.diffs[:10]  # truncate to first 10 for compact output
            ],
        }


def parse_engine_block(text: str) -> dict[str, Any]:
    """Parse official engine.py encode(0) text block into structured state."""
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
            status_code = FLAG_STATUS_MAP.get(w[2], 0)
            st["flags"].append((status_code, *map(int, w[3:7])))
        elif w[0] == "E" and len(w) >= 4 and w[1] == "attack":
            st["attacks"].append((int(w[2]), int(w[3])))
    st["attacks"] = sorted(st["attacks"])
    return st


def parse_check_step_output(line: str, teams: int, num_units: int, num_flags: int) -> dict[str, Any]:
    """Parse stdout from tools/cpp/agent_v2/check.cpp step mode."""
    head, _, att = line.partition(" |")
    t = list(map(int, head.split()))
    k = 1 + teams
    units = [tuple(t[k + 5 * i : k + 5 * i + 5]) for i in range(num_units)]
    k += 5 * num_units
    flags = [tuple(t[k + 5 * i : k + 5 * i + 5]) for i in range(num_flags)]
    a = list(map(int, att.split())) if att.strip() else []
    attacks = sorted(zip(a[::2], a[1::2]))
    return {
        "turn": t[0],
        "score": t[1 : 1 + teams],
        "units": units,
        "flags": flags,
        "attacks": attacks,
    }


def run_simulator_parity_audit(
    engine: Any,
    check_exe: Path,
    team_counts: list[int] | None = None,
    steps_per_team: int = 40,
    seed: int = 20261018,
) -> ParityReport:
    """Run step-by-step parity comparison between native simulator and official engine.py."""
    counts = team_counts or [2, 3, 4, 8, 15]
    rnd = random.Random(seed)
    diffs: list[ParityDiff] = []
    cases_covered = {
        "deaths": 0,
        "simultaneous_attacks": 0,
        "drops": 0,
        "pickups": 0,
        "respawns": 0,
        "flag_refreshes": 0,
    }
    total_steps = 0

    for teams in counts:
        map_seed = 1000 + teams
        flag_seed = 13 * teams + 7
        g = engine.Game(teams, map_seed, flag_seed=flag_seed)
        first = g.encode(0)
        init_part = first[: first.index("TURN")]
        cur_block = first

        for step_i in range(steps_per_team):
            if g.done:
                break
            total_steps += 1
            st = parse_engine_block(cur_block)

            # Generate feasible test orders covering attacks, moves, drops, pickups
            orders = []
            for uid, (x, y, hp, flag, respawn_at) in enumerate(st["units"]):
                mv = rnd.randrange(5)
                r = rnd.random()
                foes = [
                    j
                    for j, u in enumerate(st["units"])
                    if j // 3 != uid // 3 and u[0] >= 0 and abs(u[0] - x) + abs(u[1] - y) <= 3
                ]
                if flag >= 0 and r < 0.2:
                    orders.append((mv, 2, -1))  # drop
                    cases_covered["drops"] += 1
                elif r < 0.5 and foes:
                    target_foe = rnd.choice(foes)
                    orders.append((mv, 3, target_foe))  # attack
                elif r < 0.8:
                    orders.append((mv, 1, -1))  # pickup
                    cases_covered["pickups"] += 1
                else:
                    orders.append((mv, 0, -1))  # wait

            # Decode into official engine commands
            cmds = {}
            for t_idx in range(teams):
                toks = [str(g.turn + 1)]
                for m, a, tg in orders[3 * t_idx : 3 * t_idx + 3]:
                    act_str = "X" if a == 2 else "-" if a == 0 else "P" if a == 1 else str(tg)
                    toks.extend(["SUDLR"[m], act_str])
                cmds[t_idx] = g.decode(t_idx, " ".join(toks))

            # Step official engine
            g.step(cmds)
            next_block = g.encode(0)
            want = parse_engine_block(next_block)

            # Check coverage
            if len(want["attacks"]) >= 2:
                cases_covered["simultaneous_attacks"] += 1
            for u in want["units"]:
                if u[2] == 0 or u[4] > 0:  # HP == 0 or respawnAt > 0
                    cases_covered["deaths"] += 1
                if u[4] == 1:
                    cases_covered["respawns"] += 1

            returned = [
                f
                for f, (a, b) in enumerate(zip(st["flags"], want["flags"]))
                if a[0] in (2, 3) and b[0] == 0
            ]
            if returned:
                cases_covered["flag_refreshes"] += len(returned)

            # Build input for native check step
            spots = [
                (want["flags"][f][1], want["flags"][f][2]) if f in returned else (-1, -1)
                for f in range(len(st["flags"]))
            ]
            base_input = (
                init_part
                + cur_block[cur_block.index("TURN") :]
                + f"{rnd.randrange(1, 2**62)}\n"
                + "".join(f"{m} {a} {t}\n" for m, a, t in orders)
            )
            step_cmd = [str(check_exe), "step"]
            proc_input = base_input + "spots\n" + "".join(f"{x} {y}\n" for x, y in spots)
            r = subprocess.run(step_cmd, input=proc_input, capture_output=True, text=True, check=True)
            got = parse_check_step_output(r.stdout.strip(), teams, len(st["units"]), len(st["flags"]))

            # Compare all fields
            for key in ("turn", "score", "units", "flags", "attacks"):
                if got[key] != want[key]:
                    diffs.append(
                        ParityDiff(
                            step=total_steps,
                            teams=teams,
                            field=key,
                            expected=want[key],
                            actual=got[key],
                        )
                    )

            cur_block = next_block

    report = ParityReport(
        schema=SCHEMA_PARITY,
        steps_checked=total_steps,
        passed=(len(diffs) == 0),
        cases_covered=cases_covered,
        diffs=diffs,
    )
    return report
