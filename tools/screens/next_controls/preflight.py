"""Eight-worker host preflight for the frozen next-controls arms (T0077/T0078).

    python tools/repo/task.py lock cpu --task T0077 -- python tools/screens/next_controls/preflight.py --variants-config ... --seed ... --out reports/host/agent_v2_rl_acceptance/PREFLIGHT.json

16 maps at each of 2/5/8/15 teams, eight workers, 2000/50 ms reply limits.
Gates: first < 1800 ms, steady < 48 ms, no all-seat faults/late/dead, complete telemetry.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "bots" / "p_only" / "tools"))

from tools.screens.next_controls import telemetry  # noqa: E402

WORKERS = 8
COUNTS = [2, 5, 8, 15]
MAPS = 16
LIMITS = (2.0, 0.050)
STEADY_MAX_MS, FIRST_MAX_MS = 48.0, 1800.0
ROSTER = REPO / "profiles" / "meow_rl_screen.json"


def _failed_job(job, error: str) -> dict:
    fx, name, entry, fmt, commands, info, root = job
    return {
        "variant": name, "fixture_id": fx["fixture_id"], "teams": fx["teams"], "seat": fx["seat"],
        "error": error, "turns_played": None, "runner_turns": None,
        "faults": None, "dead": None, "late": None,
        "first_ms": None, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None,
        "opponent_faults": None, "opponent_late": None, "opponent_dead": None, "opponent_max_ms": None,
        "telemetry": telemetry.parse(fmt, []),
    }


def run_job(job):
    from fusion import runner  # noqa: PLC0415
    from roster.strength import utility  # noqa: PLC0415
    from training.collect import Fixture, play  # noqa: PLC0415

    fx, name, entry, fmt, commands, info, root = job
    original_bot = runner.Bot

    class TelemetryBot(original_bot):
        def __init__(self, *args, **kwargs):
            self.identity_line = None
            super().__init__(*args, **kwargs)

        def _read_stderr(self):
            # The runner's bounded tail otherwise discards the first-turn identity.
            try:
                while True:
                    line = self.proc.stderr.readline(4096)
                    if not line:
                        break
                    line = line.rstrip()
                    if self.identity_line is None and telemetry.IDENTITY_RE.search(line):
                        self.identity_line = line
                    self.stderr.append(line)
            except (OSError, ValueError):
                pass

        def report(self):
            report = super().report()
            if self.identity_line is not None:
                report["stderr_tail"] = [self.identity_line, *report["stderr_tail"]]
            return report

    seats = list(fx["opponents"])
    seats.insert(fx["seat"], "@candidate")
    fixture = Fixture(f"{fx['fixture_id']}-{name}", fx["teams"], fx["map_seed"], fx["flag_seed"], seats)
    t0 = time.time()
    try:
        runner.Bot = TelemetryBot
        match = play((fixture, commands | {"@candidate": [str(REPO / entry["binary"])]},
                      info | {"@candidate": {"family": fmt, "sha256": entry["binary_sha256"]}}, root, LIMITS))
    except Exception as exc:
        return _failed_job(job, f"{type(exc).__name__}: {exc}")
    finally:
        runner.Bot = original_bot

    me = match["bots"][fx["seat"]]
    others = [b for i, b in enumerate(match["bots"]) if i != fx["seat"]]
    replies = [turn[fx["seat"]] for turn in match["replies"]]
    parsed = telemetry.parse(fmt, me.get("stderr_tail", []))
    parsed["runner_turns"] = sum(reply is not None for reply in replies)
    return {
        "variant": name, "fixture_id": fx["fixture_id"], "teams": fx["teams"], "seat": fx["seat"],
        "wall_s": round(time.time() - t0, 2),
        "turns_played": len(match["blocks"]), "runner_turns": parsed["runner_turns"],
        "score": match["result"]["score"], "utility": utility(match["result"]["score"], fx["seat"]),
        "faults": me["faults"], "dead": me["dead"], "late": me["late"],
        "first_ms": me["first_ms"] if replies and replies[0] is not None else None,
        "p50_ms": me["p50_ms"], "p95_ms": me["p95_ms"], "p99_ms": me["p99_ms"], "max_ms": me["max_ms"],
        "fault_reasons": [f for f in match["faults"] if f[1] == fx["seat"]][:5],
        "opponent_faults": sum(b["faults"] for b in others),
        "opponent_late": sum(b["late"] for b in others),
        "opponent_dead": sum(bool(b["dead"]) for b in others),
        "opponent_max_ms": _timing_max(others, "max_ms"),
        "telemetry": parsed, "stderr_tail": me.get("stderr_tail", []),
    }


def _timing_max(rows: list[dict], key: str) -> float | None:
    vals = [r.get(key) for r in rows]
    if not vals or any(v is None or not math.isfinite(v) for v in vals):
        return None
    return round(max(vals), 3)


def verdict(rows: list[dict]) -> tuple[bool, list[str]]:
    why = []
    if not rows:
        return False, ["no games"]
    errors = [r for r in rows if r.get("error")]
    if errors:
        why.append(f"runner errors in {len(errors)} games")
    for key in ("faults", "late", "opponent_faults", "opponent_late", "opponent_dead"):
        values = [r.get(key) for r in rows]
        if any(v is None for v in values):
            why.append(f"{key} missing")
        total = sum(int(v) for v in values if v is not None)
        if total:
            why.append(f"{key} {total}")
    if any(r.get("dead") for r in rows):
        why.append(f"dead {sum(bool(r.get('dead')) for r in rows)}")
    for key in ("refused", "inconsistent"):
        values = [((r.get("telemetry") or {}).get("counters") or {}).get(key) for r in rows]
        total = sum(v for v in values if v is not None)
        if total:
            why.append(f"{key} blocks {total}")
    missing = []
    for r in rows:
        parsed = r.get("telemetry") or {}
        fmt = parsed.get("format")
        if fmt not in telemetry.FORMATS:
            missing.append(r["fixture_id"])
        elif not telemetry.complete(fmt, parsed, r.get("turns_played"))[0]:
            missing.append(r["fixture_id"])
    if missing:
        why.append(f"telemetry missing or short in {len(missing)} games")
    for key, label, limit in (("max_ms", "steady", STEADY_MAX_MS), ("first_ms", "first", FIRST_MAX_MS)):
        values = [r.get(key) for r in rows]
        present = [v for v in values if v is not None and math.isfinite(v)]
        if len(present) != len(values):
            why.append(f"{label} timing missing")
        if present and max(present) >= limit:
            why.append(f"{label} reply {max(present)} ms >= {limit}")
    return not why, why


def turn_coverage(fmt: str, rows: list[dict]) -> dict:
    """Coverage over games with both observed and game turn counts; missing stays missing."""
    pairs = []
    for r in rows:
        parsed = r.get("telemetry") or {}
        seen = ((parsed.get("counters") or {}).get("turns") if fmt == "agent_v2"
                else parsed.get("runner_turns"))
        expected = r.get("turns_played")
        if seen is not None and expected is not None:
            pairs.append((seen, expected))
    seen = sum(p[0] for p in pairs) if pairs else None
    expected = sum(p[1] for p in pairs) if pairs else None
    return {"source": "agent_v2.counters.turns" if fmt == "agent_v2" else "runner.replies",
            "games": len(pairs), "reported_turns": seen, "game_turns": expected,
            "fraction": round(seen / expected, 4) if expected else None}


def host_info() -> dict:
    def run(cmd):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    return {
        "node": platform.node(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "power_scheme": run(["powercfg", "/getactivescheme"]) if sys.platform == "win32" else None,
        "gpu": run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader"]),
        "gpu_processes": run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]),
        "threads_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants-config", required=True, help="load_variants-format list, including optional include lists")
    ap.add_argument("--seed", type=int, required=True, help="fresh seed reserved for this preflight")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--ai-test", default=None, help="path to AI_Test checkout")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.workers != WORKERS:
        raise SystemExit(f"the gate runs on exactly {WORKERS} workers; {a.workers} is refused")
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a preflight is never overwritten")

    from bots.p_only.tools.build import load_variants  # noqa: PLC0415
    from fusion import runner  # noqa: PLC0415
    from fusion.paths import ai_test_root, bot_path, engine_path, git_commit, load_engine, load_pool, sha256  # noqa: PLC0415
    from roster.strength import schedule  # noqa: PLC0415

    config_path = Path(a.variants_config).resolve()
    config_sha = sha256(config_path)
    specs = load_variants(config_path)
    if not specs:
        raise SystemExit("variants config contains no arms")
    manifest_path = REPO / "build" / "p_only" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest {manifest_path} not found: run build first")
    built = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = built["variants"]
    variants = {}
    for spec in specs:
        name = spec["name"]
        if name in variants:
            raise SystemExit(f"variant {name} is defined twice")
        if name not in manifest:
            raise SystemExit(f"{name} is not in manifest: build it first")
        entry = manifest[name]
        if entry.get("source") != spec["source"]:
            raise SystemExit(f"{name}: manifest source differs from variants config; rebuild it")
        for kind in ("bundle", "binary"):
            path = REPO / entry[kind]
            if not path.is_file() or sha256(path) != entry[f"{kind}_sha256"]:
                raise SystemExit(f"{name}: {kind} missing or differs from manifest; rebuild it")
        variants[name] = {"source": spec["source"], "role": spec.get("role", ""),
                          "telemetry": telemetry.format_for(spec), "bundle": entry["bundle"],
                          "bundle_sha256": entry["bundle_sha256"], "binary_sha256": entry["binary_sha256"]}

    root = ai_test_root(a.ai_test)
    os.environ["AI_TEST_ROOT"] = str(root)
    roster = json.loads(ROSTER.read_text(encoding="utf-8"))["opponents"]["reference"]
    sched = schedule(MAPS, COUNTS, roster, a.seed, root)
    jobs = [(fx, name, manifest[name], v["telemetry"], {}, {}, root)
            for fx in sched["fixtures"] for name, v in variants.items()]
    print(f"[next-controls preflight] {len(variants)} variants x {len(sched['fixtures'])} fixtures = {len(jobs)} games, "
          f"{WORKERS} workers, seed {a.seed}", flush=True)
    if a.dry_run:
        return 0

    pool = {e["name"]: e for e in load_pool()["bots"]}
    commands, info = {}, {}
    for ref in sched["reference"]:
        path = bot_path(pool[ref["name"]], root)
        if sha256(path) != ref["source_sha256"]:
            raise SystemExit(f"reference bot {ref['name']} changed since scheduling")
        commands[ref["name"]] = runner.command_for(path, REPO / "build" / "roster")
        info[ref["name"]] = {"family": pool[ref["name"]]["family"], "sha256": ref["source_sha256"]}
    info |= {"__engine__": sha256(engine_path(root)), "__commit__": git_commit(root)}
    load_engine(root)
    jobs = [(fx, name, entry, fmt, commands, info, root) for fx, name, entry, fmt, _, _, root in jobs]
    before = host_info()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    t0 = time.time()
    rows = []
    try:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            for row in ex.map(run_job, jobs):
                rows.append(row)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        rows.extend(_failed_job(job, error) for job in jobs[len(rows):])
    after = host_info()

    summary, ok_all = {}, True
    for name, variant in variants.items():
        rs = [r for r in rows if r["variant"] == name]
        ok, why = verdict(rs)
        if any(sum(r["teams"] == c for r in rs) != MAPS for c in COUNTS):
            ok = False
            why.append("fixture count differs from 16 per count")
        ok_all &= ok
        coverage = turn_coverage(variant["telemetry"], rs)
        by_count = {str(c): {label: _timing_max([r for r in rs if r["teams"] == c], key)
                             for label, key in (("steady_max_ms", "max_ms"), ("first_max_ms", "first_ms"),
                                                ("p99_max_ms", "p99_ms"), ("p95_max_ms", "p95_ms"),
                                                ("p50_max_ms", "p50_ms"))} for c in COUNTS}
        summary[name] = {**variant, "games": len(rs), "passed": ok, "why_not": why, "by_count": by_count,
                         "work": telemetry.work_totals(variant["telemetry"], [r["telemetry"] for r in rs]),
                         "turn_coverage": coverage["fraction"], "turn_coverage_details": coverage}
        print(f"[next-controls preflight] {name}: {'PASS' if ok else 'FAIL'} {why}", flush=True)

    receipt = {
        "schema": "ctf-next-controls-preflight-1", "generator": "tools/screens/next_controls/preflight.py",
        "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "wall_s": round(time.time() - t0, 1),
        "seed": a.seed, "workers": WORKERS, "counts": COUNTS, "maps_per_count": MAPS,
        "limits_ms": [1000 * LIMITS[0], 1000 * LIMITS[1]],
        "gates_ms": {"steady": STEADY_MAX_MS, "first": FIRST_MAX_MS},
        "roster": roster, "roster_path": ROSTER.relative_to(REPO).as_posix(), "roster_sha256": sha256(ROSTER),
        "variants_config": config_path.relative_to(REPO).as_posix() if config_path.is_relative_to(REPO) else str(config_path),
        "variants_config_sha256": config_sha,
        "schedule_fixtures": len(sched["fixtures"]), "reference": sched["reference"],
        "build": {k: built.get(k) for k in ("compiler", "flags", "host")},
        "host_before": before, "host_after": after,
        "code": {"repo_commit": git_commit(REPO), "engine_sha256": info["__engine__"]},
        "passed": ok_all, "variants": summary, "games": rows,
    }
    text = json.dumps(receipt, indent=1, allow_nan=False) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with out.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    except FileExistsError:
        raise SystemExit(f"{out} exists: a preflight is never overwritten")
    print(f"[next-controls preflight] {'PASSED' if ok_all else 'BLOCKED'} at {WORKERS} workers; receipt {out}", flush=True)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
