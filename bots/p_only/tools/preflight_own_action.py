"""Official host preflight for T0051 (host only).

Tests the 4 arms across a concurrency ladder (1, 2, 4, 6 workers) and representative load
(counts 2, 5, 10, 15) under official limits (2000 ms initialization, 50 ms subsequent replies).
Retains first/regular-turn p50/p99/max latencies, late/dead/fault records, and DSCORE stderr counters.

A level passes only with no dead process, late reply, fault, DSCORE deadline drop or invalid fallback, a DSCORE
line from every scoring arm, and an n=15 maximum reply under 48 ms. The admitted concurrency is the largest level of
an unbroken run of passing levels from the smallest; when the smallest level fails nothing is admitted (0) and the
script exits 1. Run it, and commit its receipt, before the screen: a later run reproduces a decision, it does not
make one. --seed is required and must be a seed reserved for the preflight in docs/DIGEST.md, never the screen's
evaluation seed (roster.strength.schedule derives maps from (seed, count, map), so the same seed replays the
screen's first map of each count).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from fusion import matchlog, runner
from fusion.paths import ai_test_root, bot_path, engine_path, git_commit, load_engine, load_pool, sha256
from roster.strength import REFERENCE, schedule, utility
from training.collect import Fixture, play
from roster import trajectory

DSCORE_RE = re.compile(r"DSCORE\s+roots\s+(\d+)\s+scored\s+(\d+)\s+switched\s+(\d+)\s+no_scores\s+(\d+)\s+k1\s+(\d+)\s+deadline\s+(\d+)\s+invalid\s+(\d+)\s+candidates\s+(\d+)\s+unsupported\s+(\d+)")

def parse_dscore(stderr_lines: list[str]) -> dict | None:
    for line in reversed(stderr_lines):
        m = DSCORE_RE.search(line)
        if m:
            return {
                "roots": int(m.group(1)),
                "scored": int(m.group(2)),
                "switched": int(m.group(3)),
                "no_scores": int(m.group(4)),
                "k1": int(m.group(5)),
                "deadline": int(m.group(6)),
                "invalid": int(m.group(7)),
                "candidates": int(m.group(8)),
                "unsupported": int(m.group(9)),
            }
    return None

def run_job(job):
    fx, arm, manifest_entry, commands, info, root, limits = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{arm}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    
    # Candidate command
    cand_cmd = [manifest_entry["binary"]]
    cand_info = {"family": "p2", "sha256": manifest_entry["binary_sha256"]}
    all_commands = dict(commands)
    all_commands["@candidate"] = cand_cmd
    all_info = dict(info)
    all_info["@candidate"] = cand_info
    
    t0 = time.time()
    match = play((fixture, all_commands, all_info, root, limits))
    wall_s = round(time.time() - t0, 2)
    
    score = match["result"]["score"]
    me = match["bots"][fx["seat"]]
    dscore = parse_dscore(me.get("stderr_tail", []))
    
    return {
        "arm": arm,
        "fixture_id": fx["fixture_id"],
        "teams": fx["teams"],
        "seat": fx["seat"],
        "wall_s": wall_s,
        "score": score,
        "utility": utility(score, fx["seat"]),
        "faults": me["faults"],
        "dead": me["dead"],
        "late": me["late"],
        "first_ms": me["first_ms"],
        "avg_ms": me.get("avg_ms"),
        "p50_ms": me["p50_ms"],
        "p95_ms": me["p95_ms"],
        "p99_ms": me["p99_ms"],
        "max_ms": me["max_ms"],
        "dscore": dscore,
        "stderr_tail": me.get("stderr_tail", [])[-3:],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True,
                        help="preflight schedule seed: reserved for the preflight in docs/DIGEST.md, never the evaluation seed")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 6], help="Concurrency ladder to test")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    root = ai_test_root()
    os.environ["AI_TEST_ROOT"] = str(root)
    
    manifest_path = REPO / "build" / "p_only" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["variants"]
    
    arms = [
        "v12_ponly_xxl",
        "v12_ponly_xxl_aa",
        "v12_ponly_xxl_first_trial_linear_s101",
        "v12_ponly_xxl_first_trial_mlp_s101",
    ]
    for arm in arms:
        if arm not in manifest:
            raise SystemExit(f"Arm {arm} not in manifest")
    
    # 1 map each at counts 2, 5, 10, 15
    teams_list = [2, 5, 10, 15]
    seed = args.seed
    sched = schedule(1, teams_list, REFERENCE, seed, root)
    fixtures = sched["fixtures"]
    
    # Build reference commands
    pool = {e["name"]: e for e in load_pool()["bots"]}
    build = REPO / "build" / "roster"
    commands, info = {}, {}
    for ref in sched["reference"]:
        path = bot_path(pool[ref["name"]], root)
        commands[ref["name"]] = runner.command_for(path, build)
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}
    load_engine(root)
    
    limits = (2.0, 0.050)
    
    jobs = []
    for fx in fixtures:
        for arm in arms:
            jobs.append((fx, arm, manifest[arm], commands, info, root, limits))
            
    print(f"[PREFLIGHT] Testing {len(jobs)} jobs across workers {args.workers} (seed {seed})", flush=True)
    if args.dry_run:
        return

    results_by_worker = {}
    
    for w in args.workers:
        print(f"\n--- Testing concurrency: {w} workers ---", flush=True)
        t_start = time.time()
        with ProcessPoolExecutor(max_workers=w) as executor:
            batch_results = list(executor.map(run_job, jobs))
        total_time = round(time.time() - t_start, 1)
        
        # Analyze results
        total_games = len(batch_results)
        total_faults = sum(int(r.get("faults") or 0) for r in batch_results)
        total_late = sum(int(r.get("late") or 0) for r in batch_results)
        total_dead = sum(1 for r in batch_results if r.get("dead"))
        
        n15_results = [r for r in batch_results if r["teams"] == 15]
        n15_max = max(r["max_ms"] for r in n15_results) if n15_results else 0.0
        n15_p99 = max(r["p99_ms"] for r in n15_results) if n15_results else 0.0
        
        # Check deadline counters in DSCORE
        scoring_results = [r for r in batch_results if r["dscore"] is not None]
        total_deadline_hits = sum(r["dscore"]["deadline"] for r in scoring_results)
        total_switches = sum(r["dscore"]["switched"] for r in scoring_results)
        total_scored = sum(r["dscore"]["scored"] for r in scoring_results)
        
        # Strict gating: 0 dead processes, 0 late replies, 0 faults, 0 deadline drops,
        # all scoring hooks reported valid DSCORE with 0 invalid fallbacks, and n15 max under 48ms
        has_invalid = any(r["dscore"] is not None and r["dscore"]["invalid"] > 0 for r in batch_results)
        missing_dscore = any(r["dscore"] is None and "first_trial" in r["arm"] for r in batch_results)
        safe_pass = (
            total_dead == 0
            and total_late == 0
            and total_faults == 0
            and total_deadline_hits == 0
            and not has_invalid
            and not missing_dscore
            and n15_max < 48.0
        )

        worker_summary = {
            "workers": w,
            "total_games": total_games,
            "wall_seconds": total_time,
            "total_late_replies": total_late,
            "total_dead_processes": total_dead,
            "total_faults": total_faults,
            "n15_max_latency_ms": n15_max,
            "n15_p99_latency_ms": n15_p99,
            "scoring_dscore_summary": {
                "total_scored_roots": total_scored,
                "total_switched_roots": total_switches,
                "total_deadline_drop_counter": total_deadline_hits,
            },
            "safe_pass": safe_pass,
            "results": batch_results,
        }
        results_by_worker[w] = worker_summary
        print(f"[PREFLIGHT SUMMARY w={w}] wall={total_time}s late={total_late} dead={total_dead} "
              f"n15_p99={n15_p99}ms n15_max={n15_max}ms deadline_hits={total_deadline_hits} "
              f"SAFE_PASS={worker_summary['safe_pass']}", flush=True)
    
    # Admitted concurrency: the largest level of the unbroken passing run from the smallest; none passing admits 0
    admitted_workers = 0
    for w in sorted(results_by_worker):
        if not results_by_worker[w]["safe_pass"]:
            break
        admitted_workers = w
    
    print(f"\n[PREFLIGHT DECISION] Admitted concurrency: {admitted_workers} workers"
          + ("" if admitted_workers else " (no safe configuration: stop, do not run the screen)"), flush=True)
    
    receipt = {
        "schema": "ctf-fusion-host-preflight-1",
        "task": "T0051",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "desktop-g60ojud",
        "generator": "bots/p_only/tools/preflight_own_action.py",
        "seed": seed,
        "official_limits": {"first_ms": 2000, "turn_ms": 50},
        "admitted_workers": admitted_workers,
        "concurrency_ladder": {
            str(w): {k: v for k, v in summary.items() if k != "results"}
            for w, summary in results_by_worker.items()
        },
        "detailed_runs": {
            str(w): summary["results"] for w, summary in results_by_worker.items()
        }
    }
    
    out1 = REPO / "bots" / "p_only" / "report" / "own_action_first_trial" / "PREFLIGHT.json"
    out2 = REPO / "reports" / "host" / "own_action_first_trial" / "PREFLIGHT.json"
    for out in (out1, out2):                                        # atomic: a crash never leaves half a receipt
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, out)
    print(f"Preflight receipts written to {out1} and {out2}")
    if not admitted_workers:
        sys.exit(1)

if __name__ == "__main__":
    main()
