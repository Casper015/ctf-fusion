"""Paired screen of the P-only bots, end to end, from p_only/profiles/screen.json (host only).

    python p_only/tools/run_screen.py                 # the configured screen, official limits
    python p_only/tools/run_screen.py --smoke         # 2 counts x 1 map, 1 s turns, 4 arms: checks the chain (not evidence)
    python p_only/tools/run_screen.py --analyze-only  # re-run the analysis on the results so far
    python p_only/tools/run_screen.py --dry-run       # build, schedule, identities and job order; no games

Steps (re-run the same command after an interruption; finished games are kept):

  1. build     p_only/tools/build.py --stats for every arm: bundled single-file bots,
               build/p_only/manifest.json with bundle and binary SHA-256
  2. schedule  roster.strength schedule: maps_per_count x teams fixtures, frozen seed;
               opponents from the frozen reference roster
  3. run       p_only/tools/screen_runner.py: every arm (and each aa copy, the identical
               binary under another name) on every fixture, interleaved fixture by
               fixture with the arm order rotated; rows carry an identity digest of
               bundle, binary, every inlined source and model, runner code, schedule
               and limits (identities.json), plus the engine hash
  4. analyze   p_only/tools/analyze.py: paired differences and contrasts per
               team-count band with 95% intervals, re-run noise, timing and faults,
               search and value counters, completeness; rows of another identity
               are stale

Results (small JSON and Markdown, meant to be committed) land in report_dir with
RUN.json (config, commits, host, wall time per step); raw results and match logs
stay in work_dir. The analysis exits with status 2 when a fixture is missing for
any arm: rerun step 3 until it is complete, or report it as incomplete.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "p_only" / "tools"))

from fusion.paths import ai_test_root, git_commit  # noqa: E402

AA = "~aa"


def step(name: str, args: list[str], timings: dict, check: bool = True) -> int:
    print(f"[p-only screen] {name}: python {' '.join(args)}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, *args], cwd=REPO, check=check)
    timings[name] = round(time.time() - t0, 1)
    return r.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "p_only" / "profiles" / "screen.json"))
    ap.add_argument("--ai-test", help="AI_Test checkout (or AI_TEST_ROOT)")
    ap.add_argument("--smoke", action="store_true", help="tiny plumbing check with relaxed limits (not evidence)")
    ap.add_argument("--workers", type=int, help="override the configured workers")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build, schedule, identities and job order; play nothing")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    work, report = REPO / cfg["work_dir"], REPO / cfg["report_dir"]
    teams, maps, turn_ms = cfg["teams"], cfg["maps_per_count"], cfg["turn_ms"]
    arms, aa, comparisons = list(cfg["arms"]), list(cfg.get("aa", [])), [list(c) for c in cfg.get("comparisons", [])]
    contrasts = dict(cfg.get("contrasts", {}))
    if a.smoke:
        work = work / "smoke"
        report = work / "report"
        teams, maps, turn_ms = [2, 5], 1, 1000
        arms, aa = ["c08_control", "c08_control_2ply", "c08_ponly_m", "v12_ponly_m"], ["c08_control"]
        comparisons = [["c08_ponly_m", "c08_control"], ["c08_ponly_m", "c08_control_2ply"]]
        contrasts = {}
    work.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    timings, t_all = {}, time.time()
    sched, results = work / "schedule.json", work / "results.jsonl"
    manifest_path = REPO / "build" / "p_only" / "manifest.json"
    if not a.analyze_only:
        step("build", [str(REPO / "p_only" / "tools" / "build.py"), "--stats", *arms], timings)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))["variants"]
        if not sched.exists():
            sched_cmd = ["-m", "roster.strength", "schedule", "--maps-per-count", str(maps), "--teams", *map(str, teams),
                         "--seed", str(cfg["seed"]), "--out", str(sched)]
            if cfg.get("reference"):
                sched_cmd += ["--reference", *cfg["reference"]]
            step("schedule", sched_cmd, timings)
        missing = [x for x in arms + aa if x not in manifest]
        if missing:
            raise SystemExit(f"not built: {missing}")
        run = [str(REPO / "p_only" / "tools" / "screen_runner.py"), "--schedule", str(sched), "--manifest", str(manifest_path),
               "--arms", *arms, "--results", str(results), "--workers", str(a.workers or cfg["workers"]),
               "--first-ms", str(cfg["first_ms"]), "--turn-ms", str(turn_ms), "--ai-test", str(root)]
        if aa:
            run += ["--aa", *aa]
        if cfg.get("keep_matches"):
            run += ["--keep-matches", str(work / "matches")]
        if a.dry_run:
            step("dry-run", run + ["--dry-run"], timings)
            print(f"[p-only screen] dry run: {len(arms) + len(aa)} arms built and identified, schedule {sched}; no games played")
            return
        step("run", run, timings)
        shutil.copy(work / "identities.json", report / "identities.json")
        shutil.copy(sched, report / "schedule.json")
        shutil.copy(manifest_path, report / "manifest.json")
    analyze = [str(REPO / "p_only" / "tools" / "analyze.py"), "--results", str(results), "--schedule", str(sched),
               "--manifest", str(manifest_path), "--arms", *arms, "--out", str(report),
               "--identities", str(work / "identities.json")]
    if aa:
        analyze += ["--aa", *aa]
    for c in comparisons:
        analyze += ["--compare", f"{c[0]}:{c[1]}"]
    for name, terms in contrasts.items():
        analyze += ["--contrast", f"{name}={terms}"]
    if cfg.get("keep_matches"):
        analyze += ["--matches", str(work / "matches")]
    status = step("analyze", analyze, timings, check=False)
    record = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "smoke": a.smoke, "config": cfg,
              "effective": {"teams": teams, "maps_per_count": maps, "turn_ms": turn_ms, "arms": arms, "aa": aa,
                            "workers": a.workers or cfg["workers"]},
              "ctf_fusion_commit": git_commit(REPO), "ai_test_commit": git_commit(root), "host": platform.node(),
              "os": platform.platform(), "cpu_count": os.cpu_count(), "python": platform.python_version(),
              "wall_seconds": {**timings, "total": round(time.time() - t_all, 1)}, "analysis_status": status}
    (report / "RUN.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(f"[p-only screen] done in {record['wall_seconds']['total']} s -> {report}"
          + ("  (SMOKE: not evidence)" if a.smoke else "") + ("  INCOMPLETE" if status == 2 else ""))
    sys.exit(status)


if __name__ == "__main__":
    main()
