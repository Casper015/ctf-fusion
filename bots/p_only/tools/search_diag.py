"""Fixed-work diagnostics of the joint-action search experiment on recorded games (docs/SEARCH_PORTFOLIO.md).

    python tools/repo/task.py lock cpu --task T00xx -- python bots/p_only/tools/search_diag.py \\
        --matches data/p_only/search/matches --arm c08_search_a --games 60 --fixed-samples 128 \\
        --out bots/p_only/report/search_portfolio/diag

Every configuration replays the same recorded seat streams through bots/p_only/tests/replay_test.cpp (the bot's
own turn loop, no engine, no opponents' processes): C08 at exactly --fixed-samples joint samples per search call
with the turn deadline removed (PONLY_TURN_MS very large), v12 with its refinement deadline removed, so every
configuration does its whole work and differences come from the search, not from the clock.

  c08 A (current)  B (portfolio)  C (per-branch ply 2)  D (both)  diag (plays as A; compares A's shared ply-2
  predictions with each plan's own: draws that differ, aim differences, best-plan and order changes)
  v12 A, B

Reported per configuration: replies that differ from A's (share of turns), the root's prediction digests (they
must equal A's on every turn: a different digest means a branch reached the real root), and the summed PLY2,
PORTFOLIO and SEARCH counters: predictions, successor states built, cache and memo hits, alternatives, waits,
cuts. Seconds are the replay processes' wall time on this machine and say nothing about the 50 ms reply: games at
equal full-turn time measure that (bots/p_only/profiles/screen_search.json).

Streams come from a screen's kept matches (keep_matches): the designated seat of each match of --arm (selected by
its key, like bots/p_only/tools/branch_portfolio.py). Any carrier's matches give valid streams; the replayed
decisions never change the recorded states.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from branch_portfolio import arm_key, designated_seat, unit_hash  # noqa: E402
from fusion import matchlog  # noqa: E402
from fusion.runner import find_compiler  # noqa: E402

COUNTERS = re.compile(r"^(SEARCH|PLY2|PORTFOLIO) (.*)$")
PAIR = re.compile(r"(\w+) (-?[0-9.]+(?:e-?\d+)?)")


def configs(fixed: int) -> dict[str, tuple[str, list[str]]]:
    c08 = [f"-DPONLY_FIXED_SAMPLES={fixed}", "-DPONLY_TURN_MS=1000000"]
    v12 = ["-DPONLY_REFINE_MS=1000000"]
    return {"c08_A": ("C08", c08), "c08_B": ("C08", c08 + ["-DPONLY_PORTFOLIO_K=3"]), "c08_C": ("C08", c08 + ["-DPONLY_PLY2_MODE=1"]),
            "c08_D": ("C08", c08 + ["-DPONLY_PORTFOLIO_K=3", "-DPONLY_PLY2_MODE=1"]), "c08_diag": ("C08", c08 + ["-DPONLY_PLY2_MODE=2"]),
            "v12_A": ("V12", v12), "v12_B": ("V12", v12 + ["-DPONLY_PORTFOLIO_K=3"])}


def build(name: str, carrier: str, defines: list[str], out: Path) -> Path:
    exe = out / f"diag_{name}"
    cmd = [find_compiler(), "-std=c++20", "-O2", f"-I{REPO / 'inference'}", f"-I{REPO / 'bots' / 'p_only' / 'include'}",
           f"-I{REPO / 'bots' / 'p_only' / 'bots'}", "-DFUSION_TIER_XXL", "-DPONLY_STATS=1", "-DPONLY_FAMILIES", f"-DCARRIER_{carrier}",
           *defines, str(REPO / "bots" / "p_only" / "tests" / "replay_test.cpp"), "-o", str(exe)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"build {name} failed:\n{r.stderr[-3000:]}")
    return exe


def replay(exe: Path, stream: str) -> dict:
    t0 = time.perf_counter()
    r = subprocess.run([str(exe)], input=stream.encode(), capture_output=True, timeout=7200)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-2000:])
    lines = r.stdout.decode().splitlines()
    counters = {}
    for line in r.stderr.decode().splitlines():
        m = COUNTERS.match(line)
        if m:
            for k, v in PAIR.findall(m.group(2)):
                counters[f"{m.group(1).lower()}.{k}"] = float(v)
    return {"rows": [l.split(" ", 2) for l in lines[:-1]], "truncated": int(lines[-1].split()[1]), "counters": counters,
            "seconds": time.perf_counter() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--arm", required=True, help="use the matches of this screen arm (their designated seat's streams)")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--fixed-samples", type=int, default=128)
    ap.add_argument("--configs", nargs="*", help="subset of configurations (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True, help="output path without extension (.json and .md)")
    a = ap.parse_args(argv)
    key = arm_key(a.arm)
    files = [f for f in sorted(matchlog.iter_matches(a.matches)) if f.name.split(".")[0].endswith("-" + key)]
    files = sorted(files, key=lambda f: unit_hash("diag", f.name))[:a.games]
    if not files:
        raise SystemExit(f"no matches of arm {a.arm} in {a.matches}")
    streams = []
    for f in files:
        m = matchlog.load(f)
        streams.append((m["match_id"], m["teams"], matchlog.seat_stream(m, designated_seat(m))))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / "diag_build"
    work.mkdir(exist_ok=True)
    chosen = {k: v for k, v in configs(a.fixed_samples).items() if not a.configs or k in a.configs}
    results = {}
    for name, (carrier, defines) in chosen.items():
        exe = build(name, carrier, defines, work)
        with ThreadPoolExecutor(a.workers) as pool:
            results[name] = dict(zip([s[0] for s in streams], pool.map(lambda s: replay(exe, s[2]), streams)))
        print(f"[search_diag] {name}: {len(streams)} games", flush=True)
    report = {"host": platform.node(), "arm": a.arm, "games": len(streams), "fixed_samples": a.fixed_samples,
              "teams": sorted({t for _, t, _ in streams}), "configs": {}}
    for name, per in results.items():
        base = results.get(f"{name[:3]}_A", {})
        turns = changed = digest_mismatch = truncated = 0
        sums, by_teams = defaultdict(float), defaultdict(lambda: [0, 0])
        for (mid, teams, _), r in zip(streams, per.values()):
            truncated += r["truncated"]
            for k, v in r["counters"].items():
                sums[k] += v
            ref = base.get(mid)
            if ref is None:
                continue
            for x, y in zip(r["rows"], ref["rows"]):
                turns += 1
                differs = x[2] != y[2]
                changed += differs
                digest_mismatch += x[1] != y[1]
                by_teams[teams][0] += 1
                by_teams[teams][1] += differs
        report["configs"][name] = {"turns": turns, "changed_share": round(changed / turns, 4) if turns else None,
                                   "root_digest_mismatches": digest_mismatch, "truncated": truncated,
                                   "replay_seconds": round(sum(r["seconds"] for r in per.values()), 1), "counters": dict(sums),
                                   "changed_by_teams": {str(t): round(c / n, 4) for t, (n, c) in sorted(by_teams.items())}}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    lines = [f"# Fixed-work search diagnostics", "",
             f"{report['games']} recorded games (team counts {report['teams']}) of arm {a.arm}; C08 at {a.fixed_samples} samples per "
             "search call, deadlines removed. Changed: replies that differ from the same carrier's A. Root digest mismatches must be "
             "0. Replay seconds: wall time of the replay processes (run --workers at a time) on this machine, not reply latency.", "",
             "| Config | Turns | Changed | Root digest mismatches | Truncated | Ply-2 predictions | States built | View hits | Memo hits | "
             "Alternatives | Draws differ (diag) | Top changed first / final (diag) | Replay s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for name, c in report["configs"].items():
        s = c["counters"]
        pairs = s.get("ply2.diag_pairs", 0)
        searches = s.get("ply2.diag_searches", 0)
        top = (f"{s.get('ply2.diag_first_top_changed', 0) / searches:.3f} / {s.get('ply2.diag_top_changed', 0) / searches:.3f}"
               if searches else "")
        lines.append(f"| {name} | {c['turns']} | {c['changed_share']} | {c['root_digest_mismatches']} | {c['truncated']} | "
                     f"{int(s.get('ply2.predictions', 0))} | {int(s.get('ply2.view_builds', 0))} | {int(s.get('ply2.view_hits', 0))} | "
                     f"{int(s.get('ply2.memo_hits', 0))} | {int(s.get('portfolio.assignments', 0))} | "
                     f"{round(s.get('ply2.diag_move_diff', 0) / pairs, 4) if pairs else ''} | "
                     f"{top} | "
                     f"{c['replay_seconds']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
