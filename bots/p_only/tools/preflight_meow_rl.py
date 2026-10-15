"""Official host preflight for Meowmind RL screens (T0057; host only).

Tests the actor-direct bot (meow_rl_large_v_s101) and the large-V search arm (meow_rl_large_v_s101_search)
across a concurrency ladder (1, 2, 4, 6 workers) and representative load (counts 2, 5, 10, 15)
under official limits (2000 ms initialization, 50 ms subsequent replies).
Retains first/regular-turn p50/p95/p99/max latencies, late/dead/fault records, and MEOWRL stderr counters.
Freezes admitted worker count and search deadline.
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
from roster.strength import schedule, utility
from training.collect import Fixture, play

MEOWRL_RE = re.compile(r"MEOWRL\s+turns\s+(\d+)\s+refused\s+(\d+)\s+search_used\s+(\d+)\s+search_fallback\s+(\d+)\s+leaves\s+(\d+)")

def parse_meowrl(stderr_lines: list[str]) -> dict | None:
    for line in reversed(stderr_lines):
        m = MEOWRL_RE.search(line)
        if m:
            return {
                "turns": int(m.group(1)),
                "refused": int(m.group(2)),
                "search_used": int(m.group(3)),
                "search_fallback": int(m.group(4)),
                "leaves": int(m.group(5)),
            }
    return None

def run_job(job):
    fx, arm, manifest_entry, commands, info, root, limits = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{arm}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    
    cand_cmd = [manifest_entry["binary"]]
    cand_info = {"family": "meow_rl", "sha256": manifest_entry["binary_sha256"]}
    all_commands = dict(commands)
    all_commands["@candidate"] = cand_cmd
    all_info = dict(info)
    all_info["@candidate"] = cand_info
    
    t0 = time.time()
    match = play((fixture, all_commands, all_info, root, limits))
    wall_s = round(time.time() - t0, 2)
    
    score = match["result"]["score"]
    me = match["bots"][fx["seat"]]
    meowrl = parse_meowrl(me.get("stderr_tail", []))
    
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
        "meowrl": meowrl,
        "stderr_tail": me.get("stderr_tail", [])[-3:],
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261950,
                        help="preflight schedule seed: reserved in docs/MEOWMIND_RL_SCREEN_SEEDS.md")
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 6], help="Concurrency ladder to test")
    parser.add_argument("--dry-run", action="store_true", help="Print plan and exit")
    args = parser.parse_args()

    root = ai_test_root()
    os.environ["AI_TEST_ROOT"] = str(root)
    
    manifest_path = REPO / "build" / "p_only" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["variants"]
    
    arms = [
        "meow_rl_large_v_s101",
        "meow_rl_large_v_s101_search",
    ]
    for arm in arms:
        if arm not in manifest:
            raise SystemExit(f"Arm {arm} not in manifest")
    
    teams_list = [2, 5, 10, 15]
    seed = args.seed
    
    # Held-out reference roster from meow_rl_screen.json
    screen_spec = json.loads((REPO / "profiles" / "meow_rl_screen.json").read_text(encoding="utf-8"))
    ref_names = screen_spec["primary"]["reference"]
    
    sched = schedule(1, teams_list, ref_names, seed, root)
    fixtures = sched["fixtures"]
    
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
        
        total_late = sum(r["late"] for r in batch_results)
        total_dead = sum(1 for r in batch_results if r["dead"])
        total_faults = sum(r["faults"] for r in batch_results)
        max_ms_overall = max(r["max_ms"] for r in batch_results)
        max_ms_15 = max(r["max_ms"] for r in batch_results if r["teams"] == 15)
        
        arms_stats = {}
        for a in arms:
            a_res = [r for r in batch_results if r["arm"] == a]
            a_late = sum(r["late"] for r in a_res)
            a_max_ms = max(r["max_ms"] for r in a_res)
            a_p99_max = max(r["p99_ms"] for r in a_res)
            a_meowrl = [r["meowrl"] for r in a_res if r["meowrl"] is not None]
            total_refused = sum(m["refused"] for m in a_meowrl)
            total_used = sum(m["search_used"] for m in a_meowrl)
            total_fallback = sum(m["search_fallback"] for m in a_meowrl)
            arms_stats[a] = {
                "late": a_late,
                "max_ms": a_max_ms,
                "p99_max": a_p99_max,
                "refused": total_refused,
                "search_used": total_used,
                "search_fallback": total_fallback,
            }
            print(f"  [{a}] max_ms={a_max_ms} p99_max={a_p99_max} late={a_late} refused={total_refused} "
                  f"search_used={total_used} search_fallback={total_fallback}", flush=True)
        
        passed = (total_late == 0 and total_dead == 0 and total_faults == 0 and max_ms_15 < 48.0)
        
        print(f"[PREFLIGHT SUMMARY w={w}] wall={total_time}s late={total_late} dead={total_dead} "
              f"faults={total_faults} max_ms={max_ms_overall} (n=15 max={max_ms_15}) -> {'PASS' if passed else 'FAIL'}", flush=True)
              
        results_by_worker[str(w)] = {
            "workers": w,
            "wall_s": total_time,
            "passed": passed,
            "total_late": total_late,
            "total_dead": total_dead,
            "total_faults": total_faults,
            "max_ms_overall": max_ms_overall,
            "max_ms_15": max_ms_15,
            "arms": arms_stats,
            "details": batch_results,
        }

    # Admitted concurrency
    admitted_workers = 0
    for w in args.workers:
        if results_by_worker[str(w)]["passed"]:
            admitted_workers = w
        else:
            break
            
    print(f"\n[PREFLIGHT DECISION] Admitted concurrency: {admitted_workers} workers (search deadline: 30 ms)", flush=True)
    if admitted_workers == 0:
        print("[PREFLIGHT FATAL] Smallest worker level failed; cannot proceed to screening", file=sys.stderr)
        sys.exit(1)
        
    out_dir = REPO / "reports" / "host" / "meow_rl_screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "ctf-meow-rl-host-preflight-1",
        "task": "T0057",
        "seed": seed,
        "limits": limits,
        "admitted_workers": admitted_workers,
        "search_deadline_ms": 30,
        "workers_tested": args.workers,
        "results_by_worker": {w: {k: v for k, v in d.items() if k != "details"} for w, d in results_by_worker.items()},
    }
    receipt_path = out_dir / "PREFLIGHT.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Preflight receipt written to {receipt_path}")
    return admitted_workers

if __name__ == "__main__":
    main()
