"""Attach the legacy heuristic opponent forecast q_H to shard rows.

    python -m training.heuristic_baseline --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --split val test

q_H is the multi-team planner's own forecast (`multi::Planner::forecast` in
AI_Test nextgen/master.cpp: a softmax over each enemy unit's open moves with
a 12% uniform floor), printed by its tests/forecast_dump.cpp while replaying
a recorded seat stream. It is per unit; the joint q_H is the product of the
three units' marginals, which is exactly how the search samples it.

For every row (turn, team) we need an observer seat that is not `team`:
seat 0 covers teams 1..n-1 and seat 1 covers team 0. Output per shard:
qh_XXXX.npy, float32 (rows, 3, 5) move marginals in S,U,D,L,R order; NaN
where the forecast has no entry for an active unit (counted in the report);
and qh_XXXX.json, the baseline identity (AI_Test commit and the planner build
that produced it). The offline gate requires one baseline for every shard.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

from fusion import matchlog, native
from fusion.paths import NEXTGEN_REL, ai_test_root, git_commit
from fusion.protocol import DELTAS
from fusion.schema import layout as feature_layout

DELTA_INDEX = {d: i for i, d in enumerate(DELTAS)}


def parse_dump(text: str) -> dict[tuple[int, int], list[tuple[int, int, float]]]:
    out = {}
    for line in text.splitlines():
        tok = line.split()
        if len(tok) < 3:
            continue
        turn, uid = int(tok[0]), int(tok[1])
        entries = []
        for t in tok[2:]:
            if t.startswith("a"):
                continue
            xy, prob, *_ = t.split(":")
            x, y = (int(v) for v in xy.split(","))
            entries.append((x, y, float(prob)))
        out[(turn, uid)] = entries
    return out


def make_forecaster(binary: Path):
    @lru_cache(maxsize=64)
    def forecast(match_file: str, observer: int):
        match = matchlog.load(match_file)
        stream = matchlog.seat_stream(match, observer)
        # Bytes in and out: text mode would turn \n into \r\n on Windows.
        r = subprocess.run([str(binary)], input=stream.encode(), capture_output=True, timeout=600)
        if r.returncode != 0:
            raise RuntimeError(f"forecast_dump failed on {match_file}: {r.stderr.decode(errors='replace')[-500:]}")
        return parse_dump(r.stdout.decode())
    return forecast


def marginals(rows: dict, layout, matches: list[dict], forecast) -> tuple[np.ndarray, dict]:
    origin = layout.get(rows["rec"], "origin").astype(np.int64)
    out = np.full((len(rows["rec"]), 3, 5), np.nan, np.float32)
    stats = {"units": 0, "missing": 0, "off_grid": 0}
    for r in range(len(rows["rec"])):
        meta = matches[int(rows["match"][r])]
        team, turn = int(rows["team"][r]), int(rows["turn"][r])
        table = forecast(meta["file"], 1 if team == 0 else 0)
        for k in range(3):
            o = origin[r, k]
            if o < 0:
                out[r, k] = [1, 0, 0, 0, 0]
                continue
            stats["units"] += 1
            entries = table.get((turn, 3 * team + k))
            if entries is None:
                stats["missing"] += 1
                continue
            n = _size_of(meta["file"])
            ox, oy = o % n, o // n
            row = np.zeros(5)
            ok = True
            for x, y, p in entries:
                m = DELTA_INDEX.get((x - ox, y - oy))
                if m is None:
                    ok = False
                    break
                row[m] += p
            if ok:
                out[r, k] = row / row.sum()
            else:
                stats["off_grid"] += 1
    return out, stats


@lru_cache(maxsize=4096)
def _size_of(match_file: str) -> int:
    return int(matchlog.load(match_file)["init"].split("\n", 1)[0].split()[3])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-test")
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--teams", type=int, nargs="+", required=True)
    ap.add_argument("--split", nargs="+", default=["val", "test"])
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test)
    binary = native.forecast_dump(root)
    forecast = make_forecaster(binary)
    baseline = json.dumps({"planner": NEXTGEN_REL.as_posix(), "sources": native.forecast_digest(root)}, sort_keys=True)
    print(f"[heuristic] baseline {baseline} (AI_Test {git_commit(root)})", flush=True)
    layout = feature_layout()
    index = json.loads((Path(a.shards) / "index.json").read_text(encoding="utf-8"))
    for split, teams in ((s, t) for s in a.split for t in a.teams):
        d = Path(a.shards) / f"n{teams:02d}" / split
        if not d.is_dir():
            continue
        totals = {"units": 0, "missing": 0, "off_grid": 0}
        for shard in sorted(d.glob("shard_*.npz")):
            rows = dict(np.load(shard))
            qh, stats = marginals(rows, layout, index["matches"], forecast)
            target = d / shard.name.replace("shard_", "qh_").replace(".npz", ".npy")
            np.save(target, qh)
            target.with_suffix(".json").write_text(baseline, encoding="utf-8")
            totals = {k: totals[k] + stats[k] for k in totals}
        print(f"[heuristic] n{teams:02d}/{split}: {totals}", flush=True)


if __name__ == "__main__":
    main()
