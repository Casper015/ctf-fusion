"""Eight-worker host preflight gate for agent_v2 with CSP per-count P bank (T0073).

    python tools/repo/task.py lock cpu --task T0073 -- python tools/screens/preflight_agent_v2_csp.py --seed 20261960 \\
        --out reports/host/agent_v2_csp/PREFLIGHT.json

Counts: 2, 5, 8, 15
Maps per count: 16 (64 fixtures total)
Workers: 8
Limits: 2000 ms first reply, 50 ms steady reply
Gates: first reply < 1800 ms, steady reply < 48 ms, 0 faults/late/dead, 100% turn coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
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
ROSTER = REPO / "profiles" / "meow_rl_screen.json"

BASE = ("turns", "refused", "inconsistent", "search_used", "search_fallback", "leaves")
V2 = ("trajectories", "refuse_setup", "refuse_trajectory", "refuse_expand", "refuse_critic", "refuse_late",
      "max_setup_us", "max_traj_us", "max_expand_us", "max_critic_us", "max_search_us", "branches", "critic")
FIELDS = BASE + V2

COUNTER_RE = re.compile(
    r"AGENTV2\s+"
    + r"\s+".join(rf"{k}\s+(\d+)" for k in BASE)
    + r"(?:\s+"
    + r"\s+".join(rf"{k}\s+(\S+)" for k in V2)
    + ")?"
)


def parse_counters(lines: list[str]) -> dict | None:
    """The last cumulative AGENTV2 counter line of a game, or None."""
    for line in reversed(lines):
        m = COUNTER_RE.search(line)
        if m:
            vals = m.groups()
            res = {}
            for k, v in zip(BASE, vals[:len(BASE)]):
                if v is not None:
                    res[k] = int(v)
            if vals[len(BASE)] is not None:
                for k, v in zip(V2, vals[len(BASE):]):
                    if v is not None:
                        res[k] = v if k == "critic" else int(v)
            return res
    return None


def run_job(job):
    from training.collect import Fixture, play  # noqa: PLC0415
    fx, name, entry, commands, info, root = job
    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{name}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    t0 = time.time()
    cand_cmd = [str(REPO / entry["binary"])]
    match = play((fixture, dict(commands, **{"@candidate": cand_cmd}),
                  dict(info, **{"@candidate": {"family": "agent_v2", "sha256": entry["binary_sha256"]}}), root, LIMITS))
    me = match["bots"][fx["seat"]]
    others = [b for i, b in enumerate(match["bots"]) if i != fx["seat"]]
    return {
        "variant": name,
        "fixture_id": fx["fixture_id"],
        "teams": fx["teams"],
        "seat": fx["seat"],
        "wall_s": round(time.time() - t0, 2),
        "turns_played": len(match["blocks"]),
        "score": match["result"]["score"],
        "utility": utility(match["result"]["score"], fx["seat"]),
        "faults": me["faults"],
        "dead": me["dead"],
        "late": me["late"],
        "first_ms": me["first_ms"],
        "p50_ms": me["p50_ms"],
        "p95_ms": me["p95_ms"],
        "p99_ms": me["p99_ms"],
        "max_ms": me["max_ms"],
        "fault_reasons": [f for f in match["faults"] if f[1] == fx["seat"]][:5],
        "opponent_faults": sum(b["faults"] for b in others),
        "opponent_late": sum(b.get("late") or 0 for b in others),
        "opponent_dead": sum(1 for b in others if b["dead"]),
        "opponent_max_ms": max((b["max_ms"] or 0.0) for b in others),
        "counters": parse_counters(me.get("stderr_tail", [])),
        "stderr_tail": me.get("stderr_tail", [])[-2:],
    }


def verdict(rows: list[dict]) -> tuple[bool, list[str]]:
    why = []
    for key in ("faults", "late", "opponent_faults", "opponent_late", "opponent_dead"):
        total = sum(int(r[key] or 0) for r in rows)
        if total:
            why.append(f"{key} {total}")
    refused = sum((r["counters"] or {}).get("refused", 0) for r in rows)
    if refused:
        why.append(f"refused blocks {refused}")
    inconsistent = sum((r["counters"] or {}).get("inconsistent", 0) for r in rows)
    if inconsistent:
        why.append(f"inconsistent blocks {inconsistent}")
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
    return not why, why


def host_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "node": platform.node(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "power_scheme": run(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else None,
        "gpu": run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"]),
        "gpu_processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]),
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, required=True, help="fresh, reserved for this preflight in docs/MEOWMIND_RL_SCREEN_SEEDS.md")
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--variants-config", default=str(REPO / "reports" / "host" / "agent_v2_csp" / "variants.json"))
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "agent_v2_csp" / "PREFLIGHT.json"))
    ap.add_argument("--ai-test", default=None, help="path to AI_Test checkout")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    if a.workers != WORKERS:
        raise SystemExit(f"the gate runs on exactly {WORKERS} workers; {a.workers} is refused")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a preflight is never overwritten")

    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    manifest_path = REPO / "build" / "p_only" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest {manifest_path} not found: run build first")
    built = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = built["variants"]

    v_cfg = json.loads(Path(a.variants_config).read_text(encoding="utf-8"))
    variant_specs = {v["name"]: v for v in v_cfg["variants"]}

    variants = {}
    for name, spec in variant_specs.items():
        if name not in manifest:
            raise SystemExit(f"{name} is not in manifest: build it first")
        e = manifest[name]
        variants[name] = {
            "bundle": e["bundle"],
            "bundle_sha256": e["bundle_sha256"],
            "binary_sha256": e["binary_sha256"],
            "search": False,
            "critic": None,
            "role": spec.get("role", ""),
        }

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
    print(f"[agent_v2_csp preflight] {len(variants)} variants x {len(sched['fixtures'])} fixtures = {len(jobs)} games, "
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
        ok, why = verdict(rs)
        ok_all &= ok
        cs = [r["counters"] for r in rs if r["counters"]]
        by_count = {
            str(c): {
                "steady_max_ms": round(max((r["max_ms"] or 0.0) for r in rs if r["teams"] == c), 3),
                "p99_max_ms": round(max((r["p99_ms"] or 0.0) for r in rs if r["teams"] == c), 3),
                "p95_max_ms": round(max((r["p95_ms"] or 0.0) for r in rs if r["teams"] == c), 3),
                "p50_max_ms": round(max((r["p50_ms"] or 0.0) for r in rs if r["teams"] == c), 3),
                "first_max_ms": round(max((r["first_ms"] or 0.0) for r in rs if r["teams"] == c), 3),
            }
            for c in COUNTS
        }
        work = {k: sum(c.get(k, 0) for c in cs) for k in BASE}
        summary[name] = {
            **v,
            "games": len(rs),
            "passed": ok,
            "why_not": why,
            "by_count": by_count,
            "work": work,
            "turn_coverage": round(sum(c["turns"] for c in cs) / max(1, sum(r["turns_played"] for r in rs)), 4),
        }
        print(f"[agent_v2_csp preflight] {name}: {'PASS' if ok else 'FAIL'} {why}", flush=True)

    receipt = {
        "schema": "ctf-agent-v2-preflight-1",
        "generator": "tools/screens/preflight_agent_v2_csp.py",
        "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(time.time() - t0, 1),
        "seed": a.seed,
        "workers": WORKERS,
        "counts": COUNTS,
        "maps_per_count": MAPS,
        "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]],
        "gates_ms": {"steady": STEADY_MAX_MS, "first": FIRST_MAX_MS},
        "roster": roster,
        "schedule_fixtures": len(sched["fixtures"]),
        "build": {k: built.get(k) for k in ("compiler", "flags", "host")},
        "host_before": before,
        "host_after": after,
        "code": {"repo_commit": git_commit(REPO), "engine_sha256": info["__engine__"]},
        "passed": ok_all,
        "variants": summary,
        "games": rows,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, out)
    print(f"[agent_v2_csp preflight] {'PASSED' if ok_all else 'BLOCKED'} at {WORKERS} workers; receipt {out}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
