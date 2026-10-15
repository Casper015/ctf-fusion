"""Host preflight of the Meowmind RL screens: the concurrency and deadline gate (T0057; fixed by T0060; host only).

    python bots/p_only/tools/preflight_meow_rl.py --seed <reserved for the preflight> [--workers 1 2 4 6]

Plays the actor-direct bot (meow_rl_large_v_s101) and the large-V search arm (meow_rl_large_v_s101_search), as built
in build/p_only/manifest.json, on one map at each of the counts 2, 5, 10 and 15 against the screens' held-out roster
(profiles/meow_rl_screen.json), at the official limits (2000 ms first reply, 50 ms after), for each concurrency level.
The same four fixtures repeat at every level: they test load, they are not independent observations.

A level passes only if, over all its games:
  * no seat at all (ours or any opponent) faults, replies late or dies (an overloaded host shows up in opponents too);
  * every one of our games carries the bot's MEOWRL counters (a missing counter line fails the level; it is never
    read as zero);
  * our largest reply in a 15-team game stays under 48 ms.
The admitted concurrency is the largest level of the unbroken passing run from the smallest; none passing admits 0 and
exits 1. The search deadline is read from each arm's bundled source (#define MEOW_RL_SEARCH_DEADLINE_MS), not typed.

The receipt (reports/host/meow_rl_screen/PREFLIGHT.json, written atomically) records the start and end time, the seed,
the limits, each arm's bundle and binary sha256 and deadline, every game's timing, fault and counter details, and each
level's summary. Run it, and commit the receipt, before the screen: a later run reproduces a decision, it does not
make one.
"""
from __future__ import annotations

import argparse
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

from fusion import runner  # noqa: E402
from fusion.paths import ai_test_root, bot_path, engine_path, git_commit, load_engine, load_pool, sha256  # noqa: E402
from roster.strength import schedule, utility  # noqa: E402
from training.collect import Fixture, play  # noqa: E402

ARMS = ["meow_rl_large_v_s101", "meow_rl_large_v_s101_search"]
COUNTS = [2, 5, 10, 15]
LIMITS = (2.0, 0.050)
MAX_MS_15 = 48.0
MEOWRL_RE = re.compile(r"MEOWRL\s+turns\s+(\d+)\s+refused\s+(\d+)\s+search_used\s+(\d+)\s+search_fallback\s+(\d+)\s+leaves\s+(\d+)")
DEADLINE_RE = re.compile(r"^\s*#\s*define\s+MEOW_RL_SEARCH_DEADLINE_MS\s+(\d+)", re.M)
SEARCH_RE = re.compile(r"^\s*#\s*define\s+MEOW_RL_SEARCH\s+(\d+)", re.M)


def parse_meowrl(stderr_lines: list[str]) -> dict | None:
    """The last cumulative MEOWRL counter line of a game (the bot prints one every turn), or None."""
    for line in reversed(stderr_lines):
        m = MEOWRL_RE.search(line)
        if m:
            return dict(zip(("turns", "refused", "search_used", "search_fallback", "leaves"), (int(g) for g in m.groups())))
    return None


def build_settings(bundle_text: str) -> dict:
    """What the bundled source compiles to: search on/off and its deadline (the first #define is the one in force,
    the later ones sit behind #ifndef)."""
    s = SEARCH_RE.search(bundle_text)
    d = DEADLINE_RE.search(bundle_text)
    search = bool(s and int(s.group(1)))
    return {"search": search, "search_deadline_ms": int(d.group(1)) if (search and d) else None}


def level_verdict(rows: list[dict]) -> tuple[bool, list[str]]:
    """Whether one concurrency level passes, and why not."""
    why = []
    for key in ("faults", "late", "opponent_faults", "opponent_late", "opponent_dead"):
        total = sum(int(r[key] or 0) for r in rows)
        if total:
            why.append(f"{key} {total}")
    dead = sum(1 for r in rows if r["dead"])
    if dead:
        why.append(f"dead {dead}")
    missing = [r["fixture_id"] + "/" + r["arm"] for r in rows if r["meowrl"] is None]
    if missing:
        why.append(f"no MEOWRL counters in {len(missing)} games")
    m15 = max((r["max_ms"] or 0.0) for r in rows if r["teams"] == 15) if any(r["teams"] == 15 for r in rows) else 0.0
    if m15 >= MAX_MS_15:
        why.append(f"15-team max reply {m15} ms >= {MAX_MS_15}")
    return not why, why


def run_job(job):
    fx, arm, entry, commands, info, root, limits = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{arm}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    t0 = time.time()
    match = play((fixture, dict(commands, **{"@candidate": [entry["binary"]]}),
                  dict(info, **{"@candidate": {"family": "meow_rl", "sha256": entry["binary_sha256"]}}), root, limits))
    me = match["bots"][fx["seat"]]
    others = [b for i, b in enumerate(match["bots"]) if i != fx["seat"]]
    return {"arm": arm, "fixture_id": fx["fixture_id"], "teams": fx["teams"], "seat": fx["seat"],
            "wall_s": round(time.time() - t0, 2), "score": match["result"]["score"],
            "utility": utility(match["result"]["score"], fx["seat"]),
            "faults": me["faults"], "dead": me["dead"], "late": me["late"], "first_ms": me["first_ms"],
            "p50_ms": me["p50_ms"], "p95_ms": me["p95_ms"], "p99_ms": me["p99_ms"], "max_ms": me["max_ms"],
            "opponent_faults": sum(b["faults"] for b in others), "opponent_late": sum(b.get("late") or 0 for b in others),
            "opponent_dead": sum(1 for b in others if b["dead"]),
            "opponent_max_ms": max((b["max_ms"] or 0.0) for b in others),
            "meowrl": parse_meowrl(me.get("stderr_tail", [])), "stderr_tail": me.get("stderr_tail", [])[-2:]}


def atomic_json(path: Path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True, help="reserved for the preflight in docs/MEOWMIND_RL_SCREEN_SEEDS.md")
    ap.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 6])
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "meow_rl_screen" / "PREFLIGHT.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = ai_test_root()
    os.environ["AI_TEST_ROOT"] = str(root)
    manifest = json.loads((REPO / "build" / "p_only" / "manifest.json").read_text(encoding="utf-8"))["variants"]
    arms = {}
    for arm in ARMS:
        if arm not in manifest:
            raise SystemExit(f"{arm} is not built (python bots/p_only/tools/build.py --variants bots/meow_rl/variants.json)")
        e = manifest[arm]
        text = (REPO / e["bundle"]).read_text(encoding="utf-8")
        arms[arm] = {"bundle": e["bundle"], "bundle_sha256": e["bundle_sha256"], "binary_sha256": e["binary_sha256"],
                     **build_settings(text)}
    roster = json.loads((REPO / "profiles" / "meow_rl_screen.json").read_text(encoding="utf-8"))["opponents"]["reference"]
    sched = schedule(1, COUNTS, roster, a.seed, root)
    pool = {e["name"]: e for e in load_pool()["bots"]}
    commands, info = {}, {}
    for ref in sched["reference"]:
        commands[ref["name"]] = runner.command_for(bot_path(pool[ref["name"]], root), REPO / "build" / "roster")
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}
    load_engine(root)
    jobs = [(fx, arm, manifest[arm], commands, info, root, LIMITS) for fx in sched["fixtures"] for arm in ARMS]
    print(f"[preflight] {len(jobs)} games per level, workers {a.workers}, seed {a.seed}; arms {json.dumps(arms)}", flush=True)
    if a.dry_run:
        return 0
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    levels = {}
    for w in a.workers:
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=w) as ex:
            rows = list(ex.map(run_job, jobs))
        ok, why = level_verdict(rows)
        per_arm = {}
        for arm in ARMS:
            rs = [r for r in rows if r["arm"] == arm]
            counted = [r["meowrl"] for r in rs if r["meowrl"]]
            per_arm[arm] = {"games": len(rs), "games_with_counters": len(counted),
                            "max_ms": max(r["max_ms"] or 0.0 for r in rs), "p99_max": max(r["p99_ms"] or 0.0 for r in rs),
                            "first_ms_max": max(r["first_ms"] or 0.0 for r in rs),
                            **{k: sum(c[k] for c in counted) for k in ("turns", "refused", "search_used", "search_fallback", "leaves")}}
        levels[str(w)] = {"workers": w, "wall_s": round(time.time() - t0, 1), "passed": ok, "why_not": why, "arms": per_arm,
                          "games": rows}
        print(f"[preflight w={w}] {'PASS' if ok else 'FAIL'} {why} {json.dumps(per_arm)}", flush=True)
    admitted = 0
    for w in sorted(levels, key=int):
        if not levels[w]["passed"]:
            break
        admitted = int(w)
    receipt = {"schema": "ctf-meow-rl-host-preflight-2", "task": "T0057", "generator": "bots/p_only/tools/preflight_meow_rl.py",
               "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": a.seed, "counts": COUNTS,
               "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]], "roster": roster, "arms": arms, "admitted_workers": admitted,
               "levels": levels, "code": {"repo_commit": git_commit(REPO), "engine_sha256": info["__engine__"]}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(out, receipt)
    print(f"[preflight] admitted {admitted} workers; receipt {out}", flush=True)
    if not admitted:
        print("[preflight] the smallest level failed: no safe configuration, do not screen", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
