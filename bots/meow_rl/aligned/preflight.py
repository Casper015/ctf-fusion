"""The eight-worker host gate of the aligned Meowmind RL experiment (task T0061, docs/MEOWMIND_RL_ALIGNED.md; host only).

    python tools/repo/task.py lock cpu --task <T> -- python bots/meow_rl/aligned/preflight.py --seed <ledgered> \\
        --out reports/host/meow_rl_aligned/PREFLIGHT.json

Plays every distinct frozen variant of bots/meow_rl/aligned.json (deduplicated by binary sha256) on the same 64
fixtures: 16 maps at each of the counts 2, 5, 8 and 15 against the held-out screen roster, at the official limits
(2,000 ms first reply, 50 ms after), on EXACTLY eight concurrent game workers, jobs interleaved fixture by fixture so
every variant meets the same load. Seven variants give 448 games. The fixtures repeat for every variant: they test
load and deadlines, they are not independent observations and never strength evidence.

A variant passes only if, over all its games:
  * no seat at all (ours or any opponent) faults, replies late, sends a malformed reply or dies;
  * every game carries the bot's MEOWRL counters, covering every turn the runner played (missing counters fail; they
    are never read as zero);
  * its largest steady reply (turn 2 on) is below 48 ms and its largest first reply below 1,800 ms;
  * a search variant used its search on at least one turn (a search that always falls back is not the arm it names).
Admission costs are recorded, not gated: the largest trajectory and critic cost each game saw, the stage refusals
and whether they exceeded the frozen priors (bots/meow_rl/aligned.json "search"). The whole gate passes only if every
variant passes; a failure is kept, never rerun with fewer workers or without the arm (a new preflight needs a fresh
seed and new frozen settings). run_screen.py refuses a screen whose binaries are not the ones this receipt measured.

The receipt records the host (platform, CPU count, Windows power scheme, GPU activity before and after when
nvidia-smi exists, thread environment), the build flags and compiler of build/p_only/manifest.json, every variant's
bundle and binary sha256 and search settings, every game, first and steady timing separately, and the verdict.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
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

WORKERS = 8
COUNTS = [2, 5, 8, 15]
MAPS = 16
LIMITS = (2.0, 0.050)
STEADY_MAX_MS, FIRST_MAX_MS = 48.0, 1800.0
CONFIG = REPO / "bots" / "meow_rl" / "aligned.json"
ROSTER = REPO / "profiles" / "meow_rl_screen.json"
BASE = ("turns", "refused", "search_used", "search_fallback", "leaves")
V2 = ("trajectories", "refuse_trajectory", "refuse_expand", "refuse_critic", "refuse_late", "max_setup_us", "max_traj_us",
      "max_expand_us", "max_critic_us", "max_search_us", "max_resolve_us", "max_turn_us", "first_turn_us")
FIELDS = BASE + V2
COUNTER_RE = re.compile(r"MEOWRL\s+" + r"\s+".join(rf"{k}\s+(\d+)" for k in BASE)
                        + r"(?:\s+" + r"\s+".join(rf"{k}\s+(\d+)" for k in V2) + ")?")


def parse_counters(lines: list[str]) -> dict | None:
    """The last cumulative MEOWRL counter line of a game (search v2 adds its work fields), or None."""
    for line in reversed(lines):
        m = COUNTER_RE.search(line)
        if m:
            return {k: int(v) for k, v in zip(FIELDS, m.groups()) if v is not None}
    return None


def first_defines(text: str) -> dict:
    """The MEOW_RL_SEARCH* settings in force: the first #define of each (later ones sit behind #ifndef)."""
    out = {}
    for k, v in re.findall(r"^\s*#\s*define\s+(MEOW_RL_SEARCH\w*)\s+(\d+)", text, re.M):
        out.setdefault(k, int(v))
    return out


def run_job(job):
    from training.collect import Fixture, play  # noqa: PLC0415
    fx, name, entry, commands, info, root = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{name}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    t0 = time.time()
    match = play((fixture, dict(commands, **{"@candidate": [str(REPO / entry["binary"])]}),
                  dict(info, **{"@candidate": {"family": "meow_rl", "sha256": entry["binary_sha256"]}}), root, LIMITS))
    me = match["bots"][fx["seat"]]
    others = [b for i, b in enumerate(match["bots"]) if i != fx["seat"]]
    return {"variant": name, "fixture_id": fx["fixture_id"], "teams": fx["teams"], "seat": fx["seat"],
            "wall_s": round(time.time() - t0, 2), "turns_played": len(match["blocks"]), "score": match["result"]["score"],
            "utility": utility(match["result"]["score"], fx["seat"]),
            "faults": me["faults"], "dead": me["dead"], "late": me["late"], "first_ms": me["first_ms"],
            "p50_ms": me["p50_ms"], "p95_ms": me["p95_ms"], "p99_ms": me["p99_ms"], "max_ms": me["max_ms"],
            "fault_reasons": [f for f in match["faults"] if f[1] == fx["seat"]][:5],
            "opponent_faults": sum(b["faults"] for b in others), "opponent_late": sum(b.get("late") or 0 for b in others),
            "opponent_dead": sum(1 for b in others if b["dead"]),
            "opponent_max_ms": max((b["max_ms"] or 0.0) for b in others),
            "counters": parse_counters(me.get("stderr_tail", [])), "stderr_tail": me.get("stderr_tail", [])[-2:]}


def verdict(rows: list[dict], search: bool) -> tuple[bool, list[str]]:
    why = []
    for key in ("faults", "late", "opponent_faults", "opponent_late", "opponent_dead"):
        total = sum(int(r[key] or 0) for r in rows)
        if total:
            why.append(f"{key} {total}")
    refused = sum((r["counters"] or {}).get("refused", 0) for r in rows)
    if refused:                                         # blocks the bot could not observe or resolve (not search refusals)
        why.append(f"refused blocks {refused}")
    if any(r["dead"] for r in rows):
        why.append(f"dead {sum(1 for r in rows if r['dead'])}")
    missing = [r["fixture_id"] for r in rows if r["counters"] is None or r["counters"]["turns"] != r["turns_played"]]
    if missing:
        why.append(f"counters missing or short in {len(missing)} games")
    steady = max((r["max_ms"] or 0.0) for r in rows)
    first = max((r["first_ms"] or 0.0) for r in rows)
    if any(r["max_ms"] is None for r in rows):
        why.append("steady timing missing")
    if steady >= STEADY_MAX_MS:
        why.append(f"steady max reply {steady} ms >= {STEADY_MAX_MS}")
    if first >= FIRST_MAX_MS:
        why.append(f"first reply {first} ms >= {FIRST_MAX_MS}")
    if search and not sum((r["counters"] or {}).get("search_used", 0) for r in rows):
        why.append("the search never completed")
    return not why, why


def host_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"node": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "power_scheme": run(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else None,
            "gpu": run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"]),
            "gpu_processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]),
            "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="fresh, reserved for this preflight in docs/MEOWMIND_RL_SCREEN_SEEDS.md")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "meow_rl_aligned" / "PREFLIGHT.json"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.workers != WORKERS:
        raise SystemExit(f"the gate runs on exactly {WORKERS} workers; {a.workers} is refused")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a preflight is never overwritten (a new one needs a fresh seed and its own path)")
    from preflight_meow_rl import build_settings, checked_bundle  # noqa: PLC0415
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    root = ai_test_root()
    os.environ["AI_TEST_ROOT"] = str(root)
    built = json.loads((REPO / "build" / "p_only" / "manifest.json").read_text(encoding="utf-8"))
    manifest = built["variants"]
    variants, seen = {}, {}
    for name, spec in cfg["variants"].items():
        if name not in manifest:
            raise SystemExit(f"{name} is not built (python bots/p_only/tools/build.py --variants bots/meow_rl/variants_aligned.json)")
        e = manifest[name]
        text = checked_bundle(name, e, REPO)
        if e["binary_sha256"] in seen:                                      # the same binary under two names: play it once
            variants[seen[e["binary_sha256"]]]["also"].append(name)
            continue
        seen[e["binary_sha256"]] = name
        variants[name] = {"bundle": e["bundle"], "bundle_sha256": e["bundle_sha256"], "binary_sha256": e["binary_sha256"],
                          "search": bool(spec.get("critic")), "critic": spec.get("critic"), "also": [],
                          "settings": first_defines(text) | build_settings(text)}
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))["opponents"]["reference"]
    sched = schedule(MAPS, COUNTS, roster, a.seed, root)
    pool = {e["name"]: e for e in load_pool()["bots"]}
    commands, info = {}, {}
    for ref in sched["reference"]:
        commands[ref["name"]] = runner.command_for(bot_path(pool[ref["name"]], root), REPO / "build" / "roster")
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}
    load_engine(root)
    jobs = [(fx, name, manifest[name], commands, info, root) for fx in sched["fixtures"] for name in variants]
    print(f"[aligned preflight] {len(variants)} variants x {len(sched['fixtures'])} fixtures = {len(jobs)} games, "
          f"{WORKERS} workers, seed {a.seed}", flush=True)
    if a.dry_run:
        return 0
    before = host_info()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        rows = list(ex.map(run_job, jobs))
    after = host_info()
    summary, ok_all = {}, True
    for name, v in variants.items():
        rs = [r for r in rows if r["variant"] == name]
        ok, why = verdict(rs, v["search"])
        ok_all &= ok
        cs = [r["counters"] for r in rs if r["counters"]]
        by_count = {str(c): {"steady_max_ms": max((r["max_ms"] or 0.0) for r in rs if r["teams"] == c),
                             "p99_max_ms": max((r["p99_ms"] or 0.0) for r in rs if r["teams"] == c),
                             "first_max_ms": max((r["first_ms"] or 0.0) for r in rs if r["teams"] == c)} for c in COUNTS}
        work = {k: sum(c.get(k, 0) for c in cs) for k in ("turns", "refused", "search_used", "search_fallback", "leaves",
                                                           "trajectories", "refuse_trajectory", "refuse_expand",
                                                           "refuse_critic", "refuse_late")}
        worst = {k: max((c.get(k, 0) for c in cs), default=0) for k in V2 if k.startswith(("max_", "first_"))}
        prior = cfg["search"]["prior_us"].get(v["critic"] or "", {}) if v["search"] else {}
        stage = {"setup": "max_setup_us", "trajectory": "max_traj_us", "expand": "max_expand_us", "critic": "max_critic_us"}
        summary[name] = {**v, "games": len(rs), "passed": ok, "why_not": why, "by_count": by_count, "work": work, "worst_us": worst,
                         "prior_us": prior, "prior_exceeded": {k: worst[stage[k]] > us for k, us in prior.items()},
                         "turn_coverage": (sum(c["turns"] for c in cs) / max(1, sum(r["turns_played"] for r in rs)))}
        print(f"[aligned preflight] {name}: {'PASS' if ok else 'FAIL'} {why}", flush=True)
    for name, v in list(variants.items()):
        for alias in v["also"]:                                     # the identical binary, measured under `name`
            summary[alias] = {"alias_of": name, "binary_sha256": v["binary_sha256"], "passed": summary[name]["passed"]}
    receipt = {"schema": "ctf-meow-rl-aligned-preflight-1", "generator": "bots/meow_rl/aligned/preflight.py",
               "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(time.time() - t0, 1),
               "seed": a.seed, "workers": WORKERS, "counts": COUNTS, "maps_per_count": MAPS,
               "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]], "gates_ms": {"steady": STEADY_MAX_MS, "first": FIRST_MAX_MS},
               "roster": roster, "schedule_fixtures": len(sched["fixtures"]), "config_sha256": sha256(Path(a.config)),
               "build": {k: built.get(k) for k in ("compiler", "flags", "host")}, "host_before": before, "host_after": after,
               "code": {"repo_commit": git_commit(REPO), "engine_sha256": info["__engine__"]},
               "passed": ok_all, "variants": summary, "games": rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    print(f"[aligned preflight] {'PASSED' if ok_all else 'BLOCKED'} at {WORKERS} workers; receipt {out}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
