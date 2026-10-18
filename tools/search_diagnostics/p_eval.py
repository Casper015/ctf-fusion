"""Factual P fidelity on recorded games (task T0085).

P is evaluated only at factual states: the designated seat's recorded stream is replayed through the package's
native tool (diag_tool pmarg), which prints, for every opponent unit P predicts at that turn, the movement
marginals, the feasible moves and, per move, P(wait, pickup, drop, attack) and P(target slot) with the slots' unit
ids. The labels are the opponents' recorded replies of the same turn (what they actually submitted). Never an
alternative state: this is not oracle forecasting.

Labels are feasibility-projected requests, the rule P was trained on (fusion/labels.py label_match): what the unit
asked for, projected onto what P's feasibility masks (printed by the tool) allow. They are not the engine's
realized actions: the masks judge the intended destination, before any blocking. Events (each with its own
denominator):
  move    every predicted living unit with a recorded, well-formed reply; label = its submitted move, or stay when
          that move is infeasible (counted)
  act     the same units; label = wait / pickup / drop / attack after the projected move. An action the mask marks
          as without effect (a pickup off a flag cell, a drop without a flag, an attack by a carrier or an immune
          unit) is wait (counted); so is an attack whose target is neither in P's eight candidate slots nor an enemy
          within reach beyond the slot cap (counted)
  target  units whose act label is attack on a target in one of the slots; label = that slot. An attack on an enemy
          beyond the slot cap keeps its act label but has no target label (counted)
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
from fusion.labels import _hittable_beyond_cap
from fusion.protocol import ATTACK, WAIT, parse_block, parse_reply
from tools.search_diagnostics.schemas import NATIVE_P, SCHEMA_P_FIDELITY, ValidationError, finite, sha256_file, sha256_text

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
    out = score_rows(run_pmarg(tool, stream), match, tallies)
    return {"match_id": match["match_id"], "teams": match["teams"], "seat": seat, **out,
            "stream_sha256": sha256_text(stream)}


def score_rows(rows: list[dict], match: dict, tallies: dict) -> dict:
    """Score the tool's P rows of one match against its recorded replies (the label rules above)."""
    model = rows[0]["predictor"]
    size = int(match["init"].split("\n", 1)[0].split()[3])
    turns, refused = 0, []
    for row in rows:
        if "refused" in row:
            refused.append([row["turn"], row["refused"]])
            continue
        t = row["turn"]
        replies = match["replies"][t - 1]
        units = parse_block(match["blocks"][t - 1]).units
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
            m = r.move
            if u["legal"][m] != "1":
                m = 0
                tallies["move"].relabelled["infeasible move projected to stay"] += 1
            tallies["move"].add(move, m)
            act = check_distribution(u["act"][m], w + " act")
            a = r.act
            if u["act_legal"][m][a] != "1":
                a = WAIT
                tallies["act"].relabelled["effectless action scored as wait"] += 1
            slots = u["cand"]
            if a == ATTACK and r.target not in slots:
                _, _, x, y, *_ = units[uid]
                if _hittable_beyond_cap(units, r.target, team, y * size + x, size):
                    tallies["target"].excluded["enemy beyond P's slot cap (act label kept)"] += 1
                else:
                    a = WAIT
                    tallies["act"].relabelled["attack without a valid target scored as wait"] += 1
            tallies["act"].add(act, a)
            if a == ATTACK and r.target in slots:
                tgt = check_distribution(u["target"][m], w + " target")
                tallies["target"].add(tgt, slots.index(r.target))
    return {"turns": turns, "refused_turns": refused, "predictor": model}


def evaluate(tool_identity: dict, tool: Path, match_paths: list[Path], seat_name: str) -> dict:
    from tools.search_diagnostics.extract import designated_seat  # noqa: PLC0415

    if not match_paths:
        raise ValidationError("no matches to evaluate")
    by_count: dict[int, dict] = {}
    games = []
    for p in match_paths:
        match = matchlog.load(p)
        tallies = by_count.setdefault(match["teams"], {e: Tally() for e in EVENTS})
        games.append({**evaluate_match(tool, match, designated_seat(match, seat_name), tallies),
                      "match_file": p.name, "match_sha256": sha256_file(p)})
    return {
        "schema": SCHEMA_P_FIDELITY, "package": tool_identity["package"],
        "tool_binary_sha256": tool_identity["binary_sha256"], "games": games,
        "event_definitions": {
            "move": "every predicted living opponent unit with a well-formed recorded reply; label: its submitted move, stay when infeasible",
            "act": "the same units; label: wait / pickup / drop / attack after the projected move, effectless or targetless actions as wait",
            "target": "units whose act label is attack on a target in one of P's eight candidate slots; label: that slot",
        },
        "by_count": {str(n): {e: t.report() for e, t in tallies.items()} for n, tallies in sorted(by_count.items())},
        "scope": "factual states of recorded games only; P2/P3 differences here are prediction differences, not strength",
    }
