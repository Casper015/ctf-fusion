"""Run the light strength screen end to end on the host, from profiles/screen_light.json.

    python tools/run_light_screen.py                    # the configured screen (host only: official 50 ms limits)
    python tools/run_light_screen.py --smoke            # 4 games, relaxed limits, separate folder: checks the chain

Steps, each resumable (re-run the same command after an interruption):

  1. schedule   roster.strength schedule: maps_per_count maps x teams, frozen seed
  2. run        roster.strength run --candidates <admission/pool>: every candidate on every fixture,
                full logs kept for live outcomes
  3. aggregate  roster.strength aggregate --inventory --schedule: stale rows dropped
  4. live       roster.trajectory over the kept logs (every archived seat, references included)
  5. tiers      roster.tiers: above / level / below the reference field, reliability, screened admission
  6. pool       roster.registry build with the screened admission -> pool_out
  7. record     RUN.json in report_dir: config, commits, host, wall time per step, counts

Results land in report_dir (small JSON and Markdown, meant to be committed); raw
results and match logs stay in work_dir (not committed). --smoke uses 2 team
counts, 1 map, the first 2 candidates and a 1 s turn limit, writes only under
work_dir/smoke, and must not be read as strength or timing evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion.paths import ai_test_root, git_commit  # noqa: E402
from roster.strength import candidate_ids  # noqa: E402


def step(name: str, args: list[str], timings: dict) -> None:
    print(f"[light-screen] {name}: python {' '.join(args)}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)
    timings[name] = round(time.time() - t0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "profiles" / "screen_light.json"))
    ap.add_argument("--ai-test", help="AI_Test checkout (or AI_TEST_ROOT)")
    ap.add_argument("--smoke", action="store_true", help="4-game plumbing check with relaxed limits (not evidence)")
    ap.add_argument("--workers", type=int, help="override the configured workers")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    work, report = REPO / cfg["work_dir"], REPO / cfg["report_dir"]
    teams, maps, turn_ms, pool_out = cfg["teams"], cfg["maps_per_count"], cfg["turn_ms"], REPO / cfg["pool_out"]
    cands = REPO / cfg["candidates"]
    if a.smoke:
        work = work / "smoke"
        report, pool_out = work / "report", work / "report" / "open_training_pool.json"
        teams, maps, turn_ms = [2, 3], 1, 1000
        work.mkdir(parents=True, exist_ok=True)
        ids = candidate_ids(str(REPO / cfg["candidates"]))
        cands = work / "candidates.txt"
        picks = [i for i in ids if "judge/ctf/bots" in i][:1] + [i for i in ids if "new_bots/independent" in i][:1]
        cands.write_text("\n".join(picks) + "\n", encoding="utf-8")
    work.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    workers = str(a.workers or cfg["workers"])
    sched, results, strength = work / "schedule.json", work / "results.jsonl", report / "strength.json"
    timings, t_all = {}, time.time()
    if not sched.exists():
        step("schedule", ["-m", "roster.strength", "schedule", "--maps-per-count", str(maps), "--teams", *map(str, teams),
                          "--seed", str(cfg["seed"]), "--out", str(sched)], timings)
    step("run", ["-m", "roster.strength", "run", "--schedule", str(sched), "--inventory", cfg["inventory"],
                 "--candidates", str(cands), "--results", str(results), "--workers", workers,
                 "--first-ms", str(cfg["first_ms"]), "--turn-ms", str(turn_ms), "--keep-matches", str(work / "matches")], timings)
    step("aggregate", ["-m", "roster.strength", "aggregate", "--results", str(results), "--inventory", cfg["inventory"],
                       "--schedule", str(sched), "--out", str(strength)], timings)
    step("live", ["-m", "roster.trajectory", "--raw", str(work / "matches"), "--inventory", cfg["inventory"],
                  "--out", str(report / "live_profiles.json")], timings)
    step("tiers", ["-m", "roster.tiers", "--strength", str(strength), "--shortlist", cfg["shortlist"],
                   "--admission", cfg["admission"], "--out", str(report)], timings)
    step("pool", ["-m", "roster.registry", "build", "--admission", str(report / "admission_screened.json"),
                  "--assessments", cfg["assessments"], "--distance", cfg["distance"], "--inventory", cfg["inventory"],
                  "--strength", str(strength), "--out", str(pool_out)], timings)
    shutil.copy(sched, report / "schedule.json")
    s = json.loads(strength.read_text(encoding="utf-8"))
    tiers = json.loads((report / "tiers.json").read_text(encoding="utf-8"))["bots"]
    record = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smoke": a.smoke, "config": cfg,
              "effective": {"teams": teams, "maps_per_count": maps, "turn_ms": turn_ms, "workers": int(workers)},
              "ctf_fusion_commit": git_commit(REPO), "ai_test_commit": git_commit(root), "host": platform.node(),
              "os": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
              "wall_seconds": {**timings, "total": round(time.time() - t_all, 1)},
              "candidates": len(s["candidates"]), "games": sum(c["games"] for c in s["candidates"].values()),
              "stale_rows": sum((s.get("stale_rows") or {}).values()),
              "tiers": {t: sum(r["tier"] == t for r in tiers) for t in ("above", "level", "below")},
              "unreliable": [r["bot"] for r in tiers if not r["reliable"]]}
    (report / "RUN.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"[light-screen] done: {record['games']} games, tiers {record['tiers']}, {len(record['unreliable'])} unreliable, "
          f"{record['wall_seconds']['total']} s -> {report}" + ("  (SMOKE: not evidence)" if a.smoke else ""))


if __name__ == "__main__":
    main()
