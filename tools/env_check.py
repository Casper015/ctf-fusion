"""Host readiness and latency smoke test. Run it on the machine that will collect data and host the bots.

    python tools/env_check.py --ai-test C:/path/to/AI_Test
    python tools/env_check.py --ai-test ... --teams 2 6 15 --bots Astra,Sol,Fable,Player5_Omega_LIT,Player2_v005,Baseline

It answers, in reports/env_check_<host>.md and .json:
  1. Host: OS, CPU, Python, numpy, compiler, optional PyTorch/CUDA for training.
  2. Build: every listed bot compiles (-std=c++20 -O2, -static on Windows); compile times.
  3. Headless launch and protocol: one match per --teams under the real limits
     (2 s first turn, 50 ms later), through fusion/runner.py (thread-based
     pipes, Windows-safe). Every reply must parse under the engine's rules.
  4. Bot latency on this CPU: first turn vs 2.0 s; p50/p95/p99/max of turns 2-400.
  5. Simulator: inference/sim.hpp replays each match exactly.
  6. Inference cost: tools/bench_inference.cpp with the exported model (or a
     same-shaped placeholder): policy per turn vs 5 ms, value per node in us.

Do not run it on a development laptop that is busy with other work: the
numbers are only meaningful on the target host.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from fusion import matchlog, native, quant, runner  # noqa: E402
from fusion.models import PolicyNet, ValueNet  # noqa: E402
from fusion.paths import BUILD, ai_test_root, load_engine, load_json  # noqa: E402
from fusion.schema import layout as feature_layout  # noqa: E402
from training import collect, export_weights  # noqa: E402

DEFAULT_BOTS = "Astra,Sol,Fable,Player5_Omega_LIT,Player2_v005,Baseline"


def host_profile() -> dict:
    cxx = runner.find_compiler()
    version = "none"
    if cxx:
        r = subprocess.run([cxx, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        version = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "unknown"
    torch_info = "not installed"
    try:
        import torch  # noqa: PLC0415
        torch_info = f"{torch.__version__}, CUDA {'available: ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}"
    except ImportError:
        pass
    return {"host": socket.gethostname(), "os": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor() or "unknown", "logical_cpus": os.cpu_count(),
            "python": sys.version.split()[0], "numpy": np.__version__, "compiler": cxx, "compiler_version": version,
            "cxx_flags": runner.CXX_FLAGS, "torch": torch_info}


def placeholder_model(out_dir: Path) -> tuple[Path, str]:
    """Random weights with the exported shapes: inference cost does not depend on the values."""
    L = feature_layout()
    rng = np.random.default_rng(0)
    policy, value = PolicyNet(L, rng), ValueNet(L, rng)
    nets = export_weights.quantize_model(policy, value, 1.0, L)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "weights_placeholder.hpp"
    path.write_text(export_weights.header("placeholder", nets, {}, {"policy": "random", "value": "random"}), encoding="utf-8")
    return path, "fusion::models::placeholder"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-test")
    ap.add_argument("--bots", default=DEFAULT_BOTS, help="pool names; seats cycle through them")
    ap.add_argument("--teams", type=int, nargs="+", default=[2, 6])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--model", help="exported weights_<model>.hpp for the inference benchmark (default: placeholder)")
    ap.add_argument("--out", default=str(REPO / "reports"))
    a = ap.parse_args(argv)
    root = ai_test_root(a.ai_test)
    engine = load_engine(root)
    report = {"profile": host_profile(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "build": {}, "matches": [],
              "limits_ms": [1000 * x for x in getattr(engine.Game, "TIME_LIMITS", (2.0, 0.05))]}
    print(json.dumps(report["profile"], indent=1), flush=True)

    names = a.bots.split(",")
    pool = collect.resolve_pool(load_json(REPO / "profiles" / "pool.json"), root, set(names))
    commands = {}
    for n in names:
        if n not in pool:
            report["build"][n] = {"ok": False, "error": "not in pool or missing (optional bot)"}
            continue
        t0 = time.time()
        try:
            commands[n] = runner.command_for(Path(pool[n]["abs"]), REPO / "build" / "bots")
            report["build"][n] = {"ok": True, "seconds": round(time.time() - t0, 1)}
        except RuntimeError as exc:
            report["build"][n] = {"ok": False, "error": str(exc)[-500:]}
        print(f"[build] {n}: {report['build'][n]}", flush=True)
    seats_pool = [n for n in names if n in commands]
    if not seats_pool:
        raise SystemExit("no bot could be built")

    info = {n: {"family": pool[n]["family"], "sha256": "-"} for n in seats_pool}
    info["__engine__"] = "-"
    info["__commit__"] = "-"
    limits = list(getattr(engine.Game, "TIME_LIMITS", (2.0, 0.05)))
    tmp = Path(tempfile.mkdtemp(prefix="env_check_"))
    biggest = None
    for teams in a.teams:
        seats = [seats_pool[i % len(seats_pool)] for i in range(teams)]
        fx = collect.Fixture(f"env-n{teams:02d}", teams, collect.derive(a.seed, teams, "map"),
                             collect.derive(a.seed, teams, "flag", bits=63), seats)
        t0 = time.time()
        match = collect.play((fx, commands, info, str(root), limits))
        wall = time.time() - t0
        malformed = sum(1 for _, _, reason in match["faults"] if reason.startswith("malformed"))
        timeouts = sum(1 for _, _, reason in match["faults"] if reason.startswith("timeout"))
        exact, summary = native.sim_check_match(match, tmp)
        entry = {"teams": teams, "seats": seats, "wall_seconds": round(wall, 1), "score": match["result"]["score"],
                 "malformed_lines": malformed, "timeouts": timeouts, "simulator_exact": exact, "simulator": summary,
                 "bots": match["bots"]}
        report["matches"].append(entry)
        print(f"[match] {teams} teams: {wall:.0f}s, malformed {malformed}, timeouts {timeouts}, simulator {summary}", flush=True)
        for b in match["bots"]:
            print(f"   {b['name']:<20} first {b['first_ms']:>7} ms  p50 {b['p50_ms']}  p95 {b['p95_ms']}  "
                  f"p99 {b['p99_ms']}  max {b['max_ms']} ms  faults {b['faults']}  dead {b['dead']}", flush=True)
        if biggest is None or teams > biggest[0]:
            biggest = (teams, match)

    header, ns = (Path(a.model), f"fusion::models::{Path(a.model).stem.removeprefix('weights_')}") if a.model \
        else placeholder_model(BUILD / "placeholder")
    bench = native.model_tool("bench_inference", header, ns)
    r = native.run_tool(bench, matchlog.seat_stream(biggest[1], 0).encode(), args=("3",))
    report["inference"] = {"model": str(header), "teams": biggest[0], **json.loads(r.stdout.decode())}
    print(f"[inference] {report['inference']}", flush=True)

    first_limit, turn_limit = report["limits_ms"]
    bots = [b for m in report["matches"] for b in m["bots"]]
    report["verdict"] = {
        "all_bots_built": all(v["ok"] for v in report["build"].values()),
        "protocol_clean": all(m["malformed_lines"] == 0 for m in report["matches"]),
        "first_turn_within_limit": all(b["first_ms"] < first_limit for b in bots),
        "no_timeouts": all(m["timeouts"] == 0 for m in report["matches"]),
        "p99_under_40ms": all(b["p99_ms"] is not None and b["p99_ms"] < 40 for b in bots),
        "simulator_exact": all(m["simulator_exact"] for m in report["matches"]),
        "policy_p99_under_5ms": report["inference"]["policy_turn_ms"]["p99"] < 5,
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"env_check_{report['profile']['host']}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = [f"# Environment check: {report['profile']['host']}", "", f"{report['time']}", "", "## Host", ""]
    lines += [f"- {k}: {v}" for k, v in report["profile"].items()]
    lines += ["", "## Verdict", ""] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in report["verdict"].items()]
    lines += ["", "## Bot latency (turns 2-400, ms)", "", "| teams | bot | first | p50 | p95 | p99 | max | faults |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for m in report["matches"]:
        for b in m["bots"]:
            lines.append(f"| {m['teams']} | {b['name']} | {b['first_ms']} | {b['p50_ms']} | {b['p95_ms']} | {b['p99_ms']} | "
                         f"{b['max_ms']} | {b['faults']} |")
    inf = report["inference"]
    lines += ["", "## Inference", "", f"Model `{Path(inf['model']).name}` on the {inf['teams']}-team match: policy per turn "
              f"p50 {inf['policy_turn_ms']['p50']:.3f} / p99 {inf['policy_turn_ms']['p99']:.3f} ms (ceiling 5 ms); value per node "
              f"encode {inf['value_node_us']['encode']:.2f} + incremental {inf['value_node_us']['incremental']:.2f} us "
              f"(full pass {inf['value_node_us']['full']:.2f} us)."]
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[env_check] wrote {stem.with_suffix('.md')}")
    return 0 if all(report["verdict"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
