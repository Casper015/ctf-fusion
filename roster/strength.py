"""Strength screen and confirmation for candidate bots (matches: run them on the target host only).

    python -m roster.strength schedule --maps-per-count 50 --out data/roster/screen_schedule.json
    python -m roster.strength run --schedule data/roster/screen_schedule.json --inventory data/roster/bot_inventory.json \\
        --results data/roster/screen_results.jsonl --workers 8 [--keep-matches data/raw_screen]
    python -m roster.strength aggregate --results data/roster/screen_results.jsonl --out data/roster/strength.json

Every candidate plays the same frozen fixtures: per team count and map, one
seat (rotating with the map index) is the candidate's, every other seat a bot
of the frozen reference roster (default: the ten judge bots, Player5 and
Baseline, rotated deterministically). Maps, flags, seats and opponents are
identical for every candidate, so candidates compare fixture by fixture.

Each map is one independent group: one fixture per map and team count, never
"400 turns = 400 matches". The result per game is the candidate's pairwise
utility (each other team beaten 1, tied 0.5, divided by teams - 1).
aggregate reports, per team count and over the 14 counts equally weighted,
the mean utility, its standard error over maps, a 95% lower bound, faults and
reply times under the official limits (2 s first turn, 50 ms later), plus the
evidence level: "screen", or "confirmation" once a bot has at least 10,000
maps with at least 500 per team count (handoff BOT_SELECTION_PLAN.md).

Stage it: a cheap screen (tens of maps per count) for every eligible
representative, then confirmation for the finalists only. --keep-matches
stores full match logs, which double as on-policy training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from fusion import matchlog, runner
from fusion.paths import REPO, TEAM_COUNTS, ai_test_root, bot_path, git_commit, load_engine, load_pool, sha256
from roster.common import SCHEMA_STRENGTH, load_json, save_json, sha256_file
from roster.fingerprint import bots_from_inventory, resolve

REFERENCE = ["Astra", "Sol", "Fable", "Sonnet", "Opus", "DeepSeek", "Qwen", "MiMo", "Kimi", "GLM", "Player5", "Baseline"]
CONFIRM_TOTAL, CONFIRM_PER_COUNT = 10_000, 500


def derive(*parts, bits=31) -> int:
    return int.from_bytes(hashlib.sha256("/".join(map(str, parts)).encode()).digest()[:8], "little") % (1 << bits)


def schedule(maps_per_count: int, teams_list, reference: list[str], seed: int, root: Path) -> dict:
    pool = {e["name"]: e for e in load_pool()["bots"]}
    missing = [n for n in reference if n not in pool or not bot_path(pool[n], root).is_file()]
    if missing:
        raise SystemExit(f"reference bots missing from the pool or the checkout: {missing}")
    fixtures = []
    for n in teams_list:
        for m in range(maps_per_count):
            start = derive(seed, n, m, "roster") % len(reference)
            opponents = [reference[(start + i) % len(reference)] for i in range(n - 1)]
            fixtures.append({"fixture_id": f"s{seed}-n{n:02d}-m{m:05d}", "teams": n, "map_seed": derive(seed, n, m, "map"),
                             "flag_seed": derive(seed, n, m, "flag", bits=63), "seat": m % n, "opponents": opponents})
    return {"schema": SCHEMA_STRENGTH + "-schedule", "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
            "maps_per_count": maps_per_count, "team_counts": list(teams_list),
            "reference": [{"name": n, "path": pool[n]["path"], "source_sha256": sha256(bot_path(pool[n], root))} for n in reference],
            "fixtures": fixtures}


def utility(scores: list[int], seat: int) -> float:
    own = scores[seat]
    return sum(1.0 if own > s else 0.5 if own == s else 0.0 for i, s in enumerate(scores) if i != seat) / (len(scores) - 1)


def play_one(job) -> dict:
    from training.collect import Fixture, play  # noqa: PLC0415
    fx, cand, commands, info, root, limits, keep = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{cand['key']}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    match = play((fixture, commands, info, root, limits))
    score = match["result"]["score"]
    me = match["bots"][fx["seat"]]
    if keep:
        matchlog.save(Path(keep) / f"n{fx['teams']:02d}" / f"{fixture.match_id}.json.gz", match)
    return {"candidate": cand["id"], "source_sha256": cand["source_sha256"], "fixture_id": fx["fixture_id"],
            "teams": fx["teams"], "map_seed": fx["map_seed"], "seat": fx["seat"], "score": score,
            "utility": utility(score, fx["seat"]), "faults": me["faults"], "dead": me["dead"], "late": me["late"],
            "first_ms": me["first_ms"], "p99_ms": me["p99_ms"], "max_ms": me["max_ms"]}


def run(a) -> None:
    sched = load_json(a.schedule)
    root = ai_test_root(a.ai_test)
    pool = {e["name"]: e for e in load_pool()["bots"]}
    build = REPO / "build" / "roster"
    commands, info = {}, {}
    for ref in sched["reference"]:
        path = bot_path(pool[ref["name"]], root)
        if sha256(path) != ref["source_sha256"]:
            raise SystemExit(f"reference bot {ref['name']} changed since the schedule was frozen")
        commands[ref["name"]] = runner.command_for(path, build)
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    inv = load_json(a.inventory) if a.inventory else None
    cands = bots_from_inventory(inv) if inv else []
    for spec in a.bot:
        bid, _, path = spec.partition("=")
        p = resolve(path, None)
        cands.append({"id": bid, "path": str(p), "source_sha256": sha256_file(p), "binary_sha256": None})
    if a.only:
        cands = [c for c in cands if c["id"] in set(a.only)]
    out = Path(a.results)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["candidate"], r["fixture_id"]))
    limits = (a.first_ms / 1000, a.turn_ms / 1000)
    jobs = []
    for c in cands:
        c["key"] = hashlib.sha256(c["id"].encode()).hexdigest()[:10]
        cmd = runner.command_for(Path(c["path"]), build)
        cinfo = {"family": "candidate", "sha256": c["source_sha256"]}
        for fx in sched["fixtures"]:
            if (c["id"], fx["fixture_id"]) not in done:
                jobs.append((fx, c, commands | {"@candidate": cmd}, info | {"@candidate": cinfo,
                             "__engine__": sha256(root / "ctf-arena/Player7/workspace/engine.py"),
                             "__commit__": git_commit(root)}, str(root), limits, a.keep_matches))
    load_engine(root)
    print(f"[strength] {len(cands)} candidates x {len(sched['fixtures'])} fixtures; {len(jobs)} games to play, "
          f"{a.workers} workers (keep workers x teams at or below the physical cores)", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    pool_ex = ProcessPoolExecutor(a.workers) if a.workers > 1 else None
    results = pool_ex.map(play_one, jobs, chunksize=1) if pool_ex else map(play_one, jobs)
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            f.write(json.dumps(r) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"[strength] {i}/{len(jobs)}", flush=True)
    if pool_ex:
        pool_ex.shutdown()


def aggregate(rows: list[dict], teams_list=TEAM_COUNTS, z: float = 1.96) -> dict:
    by = defaultdict(lambda: defaultdict(list))
    meta = defaultdict(lambda: {"faults": 0, "dead": 0, "late": 0, "games": 0, "p99": [], "max": 0.0, "first": 0.0})
    seen = set()
    for r in rows:
        key = (r["candidate"], r["fixture_id"])
        if key in seen:
            continue  # one game per fixture and candidate: a rerun is not a new map
        seen.add(key)
        by[r["candidate"]][r["teams"]].append(r["utility"])
        m = meta[r["candidate"]]
        m["games"] += 1
        m["faults"] += r["faults"]
        m["dead"] += r["dead"] is not None
        m["late"] += r["late"]
        if r["p99_ms"] is not None:
            m["p99"].append(r["p99_ms"])
        m["max"] = max(m["max"], r["max_ms"] or 0.0)
        m["first"] = max(m["first"], r["first_ms"] or 0.0)
    out = {}
    for cand, counts in by.items():
        per = {}
        for n, u in sorted(counts.items()):
            x = np.asarray(u, float)
            se = float(x.std(ddof=1) / math.sqrt(len(x))) if len(x) > 1 else float("inf")
            per[str(n)] = {"maps": len(x), "utility": float(x.mean()), "se": se, "lcb": float(x.mean() - z * se)}
        covered = [str(n) for n in teams_list if str(n) in per]
        full = len(covered) == len(teams_list)
        mean = float(np.mean([per[n]["utility"] for n in covered])) if covered else None
        se = math.sqrt(sum(per[n]["se"] ** 2 for n in covered)) / len(covered) if covered else None
        maps = sum(p["maps"] for p in per.values())
        m = meta[cand]
        level = ("confirmation" if full and maps >= CONFIRM_TOTAL and all(per[n]["maps"] >= CONFIRM_PER_COUNT for n in covered)
                 else "screen")
        out[cand] = {"per_count": per, "team_counts_covered": len(covered), "all_counts": full, "maps": maps,
                     "utility": mean, "se": se, "lcb": None if mean is None else mean - z * se, "evidence": level,
                     "games": m["games"], "faults_per_game": m["faults"] / m["games"], "dead_games": m["dead"],
                     "late_replies": m["late"], "p99_ms_median": float(np.median(m["p99"])) if m["p99"] else None,
                     "max_ms": m["max"], "first_ms_max": m["first"]}
    return {"schema": SCHEMA_STRENGTH, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "z": z,
            "confirmation": {"total_maps": CONFIRM_TOTAL, "per_count": CONFIRM_PER_COUNT}, "candidates": out}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("schedule")
    s.add_argument("--ai-test")
    s.add_argument("--maps-per-count", type=int, default=50)
    s.add_argument("--teams", type=int, nargs="+", default=list(TEAM_COUNTS))
    s.add_argument("--reference", nargs="+", default=REFERENCE, help="frozen reference roster (pool names)")
    s.add_argument("--seed", type=int, default=20261008)
    s.add_argument("--out", default=str(REPO / "data" / "roster" / "screen_schedule.json"))
    r = sub.add_parser("run")
    r.add_argument("--ai-test")
    r.add_argument("--schedule", required=True)
    r.add_argument("--inventory")
    r.add_argument("--bot", action="append", default=[], help="extra candidate as ID=PATH")
    r.add_argument("--only", nargs="*")
    r.add_argument("--results", default=str(REPO / "data" / "roster" / "screen_results.jsonl"))
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 8))
    r.add_argument("--first-ms", type=float, default=2000.0)
    r.add_argument("--turn-ms", type=float, default=50.0)
    r.add_argument("--keep-matches", help="directory for full match logs (on-policy data)")
    g = sub.add_parser("aggregate")
    g.add_argument("--results", required=True, nargs="+")
    g.add_argument("--out", default=str(REPO / "data" / "roster" / "strength.json"))
    a = ap.parse_args(argv)
    if a.cmd == "schedule":
        out = schedule(a.maps_per_count, a.teams, a.reference, a.seed, ai_test_root(a.ai_test))
        save_json(a.out, out, indent=None)
        print(f"[strength] {len(out['fixtures'])} fixtures -> {a.out}")
    elif a.cmd == "run":
        run(a)
    else:
        rows = [json.loads(l) for p in a.results for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]
        out = aggregate(rows)
        save_json(a.out, out)
        print(f"[strength] {len(out['candidates'])} candidates -> {a.out}")


if __name__ == "__main__":
    main()
