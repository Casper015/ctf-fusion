"""Factual P fidelity on recorded games (task T0085).

P is evaluated only at factual states: the designated seat's recorded stream is replayed through the package's
native tool (diag_tool pmarg), which prints, for every opponent unit P predicts at that turn, the movement
marginals, the feasible moves and, per move, P(wait, pickup, drop, attack) and P(target slot) with the slots' unit
ids. The labels are the opponents' recorded replies of the same turn (what they actually submitted). Never an
alternative state: this is not oracle forecasting.

Events (each with its own denominator):
  move    every predicted living unit with a recorded, well-formed reply; label = its submitted move
  act     ... whose submitted move is feasible for P (P has no action head for an infeasible move: excluded, counted);
          label = wait / pickup / drop / attack; an action that P's feasibility mask (printed by the tool) marks as
          without effect after that move (a pickup off a flag cell, a drop without a flag, an attack by a carrier or
          an immune unit; the engine ignores those) is scored as wait and counted. The mask judges the intended
          destination, before any blocking
  target  ... that submitted an attack; label = the slot holding the target. A target in no slot is out of P's
          support: counted, never scored
Per event: count, mean negative log-likelihood (natural log), top-1 accuracy, expected calibration error over ten
equal-width bins of the top-1 probability, and every exclusion by reason. A label with probability 0 is counted
apart and makes the event's NLL unavailable (it is infinite), never clipped. Every probability vector must be
finite, nonnegative and sum to 1 within 1e-6; any other vector refuses the input.
"""
from __future__ import annotations

import json
import math
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from fusion import matchlog
from fusion.protocol import parse_reply
from tools.search_diagnostics.schemas import NATIVE_P, SCHEMA_P_FIDELITY, ValidationError, finite

EVENTS = ("move", "act", "target")
TOL = 1e-6
BINS = 10


def check_distribution(p, where: str) -> list[float]:
    v = finite(p, where)
    if any(x < 0 for x in v):
        raise ValidationError(f"{where}: negative probability")
    if abs(sum(v) - 1.0) > TOL:
        raise ValidationError(f"{where}: sums to {sum(v)}")
    return v


def run_pmarg(tool: Path, stream: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="diag_p_") as d:
        p = Path(d) / "stream.txt"
        p.write_text(stream, encoding="utf-8", newline="\n")
        r = subprocess.run([str(tool), "pmarg", str(p)], capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise ValidationError(f"pmarg failed: {(r.stdout or r.stderr)[:300]}")
    rows = [json.loads(x) for x in r.stdout.splitlines() if x.strip()]
    if not rows or rows[0].get("schema") != NATIVE_P:
        raise ValidationError("pmarg printed no ctf-diag-native-p-1 rows")
    return rows


class Tally:
    def __init__(self):
        self.n = 0
        self.nll = 0.0
        self.zero = 0
        self.hits = 0
        self.bins = [[0, 0.0, 0] for _ in range(BINS)]     # count, confidence sum, hits
        self.excluded: Counter = Counter()
        self.relabelled: Counter = Counter()

    def add(self, probs: list[float], label: int):
        self.n += 1
        p = probs[label]
        if p <= 0:
            self.zero += 1
        else:
            self.nll -= math.log(p)
        top = max(range(len(probs)), key=lambda i: probs[i])
        hit = int(top == label)
        self.hits += hit
        b = self.bins[min(BINS - 1, int(probs[top] * BINS))]
        b[0] += 1
        b[1] += probs[top]
        b[2] += hit

    def report(self) -> dict:
        if not self.n:
            return {"count": 0, "status": "unavailable: no supported labels", "excluded": dict(self.excluded),
                    "relabelled": dict(self.relabelled)}
        ece = sum(abs(b[1] / b[0] - b[2] / b[0]) * b[0] / self.n for b in self.bins if b[0])
        return {"count": self.n, "nll": None if self.zero else self.nll / self.n,
                "nll_status": f"unavailable: {self.zero} labels had probability 0" if self.zero else "computed",
                "zero_probability_labels": self.zero, "top1_accuracy": self.hits / self.n, "ece_10bin": ece,
                "excluded": dict(self.excluded), "relabelled": dict(self.relabelled)}


def evaluate_match(tool: Path, match: dict, seat: int, tallies: dict) -> dict:
    stream = matchlog.init_for_seat(match["init"], seat) + "".join(match["blocks"])
    rows = run_pmarg(tool, stream)
    model = rows[0]["predictor"]
    turns = 0
    for row in rows:
        if "refused" in row:
            continue
        t = row["turn"]
        replies = match["replies"][t - 1]
        turns += 1
        for u in row["units"]:
            uid, team, k = u["uid"], u["uid"] // 3, u["uid"] % 3
            w = f"{match['match_id']} turn {t} unit {uid}"
            move = check_distribution(u["move"], w + " move")
            req = parse_reply(replies[team], t)
            if req is None:
                tallies["move"].excluded["no well-formed recorded reply"] += 1
                continue
            r = req[k]
            tallies["move"].add(move, r.move)
            if u["legal"][r.move] != "1":
                tallies["act"].excluded["submitted move infeasible for P"] += 1
                continue
            act = check_distribution(u["act"][r.move], w + " act")
            label = r.act
            if u["act_legal"][r.move][label] != "1":
                label = 0                                  # the engine ignores an action without effect: a wait
                tallies["act"].relabelled["effectless action scored as wait"] += 1
            tallies["act"].add(act, label)
            if r.act != 3:
                continue
            slots = u["cand"]
            if r.target not in slots:
                tallies["target"].excluded["target outside P's candidate slots"] += 1
                continue
            tgt = check_distribution(u["target"][r.move], w + " target")
            tallies["target"].add(tgt, slots.index(r.target))
    return {"match_id": match["match_id"], "teams": match["teams"], "turns": turns, "predictor": model}


def evaluate(tool_identity: dict, tool: Path, match_paths: list[Path], seat_name: str) -> dict:
    from tools.search_diagnostics.extract import designated_seat  # noqa: PLC0415

    if not match_paths:
        raise ValidationError("no matches to evaluate")
    by_count: dict[int, dict] = {}
    games = []
    for p in match_paths:
        match = matchlog.load(p)
        tallies = by_count.setdefault(match["teams"], {e: Tally() for e in EVENTS})
        games.append(evaluate_match(tool, match, designated_seat(match, seat_name), tallies))
    return {
        "schema": SCHEMA_P_FIDELITY, "package": tool_identity["package"],
        "tool_binary_sha256": tool_identity["binary_sha256"], "games": games,
        "event_definitions": {
            "move": "every predicted living opponent unit with a well-formed recorded reply; label: its submitted move",
            "act": "those whose submitted move is feasible for P; label: wait / pickup / drop / attack, an effectless action counted and scored as wait",
            "target": "those that submitted an attack on a unit in one of P's eight candidate slots; label: that slot",
        },
        "by_count": {str(n): {e: t.report() for e, t in tallies.items()} for n, tallies in sorted(by_count.items())},
        "scope": "factual states of recorded games only; P2/P3 differences here are prediction differences, not strength",
    }
