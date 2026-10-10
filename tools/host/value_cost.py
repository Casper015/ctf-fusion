"""What a value network (and the policy's encoding) costs on the host, with real exported weights (Route A evidence).

    python tools/repo/task.py lock cpu --task T00xx -- python tools/host/value_cost.py \\
        --set inference/models_pxxl_vD --calls 7 --copies 1 6 15 --out reports/host/value_cost_DESKTOP-G60OJUD
    python tools/host/value_cost.py --set inference/models_pxxl_vD --verify     # development machines: correctness only

For each model set (--set, repeatable) the value net of --model (default general) runs in tools/cpp/value_bench.cpp
on recorded games (--streams, default the P-only fixtures with 2, 4 and 15 teams) at --calls value calls per
turn, the carriers' invocation pattern (v12: one per rollout scenario, about 7 per turn in the stage 2 screen;
C08: one per refine scenario or surviving plan, about 2.5). --copies N runs N processes at once, the load of N
busy seats. Reported per set and copy count: per-call view, encoding (v3 includes the combat summary), full
pass, incremental pass and first layer (p50/p95/p99/max microseconds; the worst copy's percentiles), per-turn
milliseconds at --calls, the first layer's share of a full pass, the inputs that change between calls (the
reuse a first-layer accumulator can exploit), weight bytes and buffer width; and the set's policy at the root:
the shared view (with the p3 combat summary), per-team encoding and the joint head, so p2 and p3 extraction
costs can be compared with real weights.

This is the V part only. Full-reply percentiles, deadline faults and the search samples V displaces come from
a timing screen of the deployed variants (bots/p_only/profiles/screen_v_pilot.json, PONLY_STATS). Timing is
evidence only from the host (AGENTS.md); --verify needs no host.
"""
from __future__ import annotations

import argparse
import gzip
import json
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import hostinfo, native  # noqa: E402
from fusion.paths import sha256  # noqa: E402
from training.header_chunks import verify_chunks  # noqa: E402

FIXTURES = REPO / "bots" / "p_only" / "tests" / "fixtures"


def run_copies(binary: Path, streams: list[bytes], calls: int, copies: int) -> list[dict]:
    procs = []
    for i in range(copies):
        p = subprocess.Popen([str(binary), str(calls)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        procs.append((p, streams[i % len(streams)]))
    def finish(pair):
        p, data = pair
        stdout, stderr = p.communicate(data)
        if p.returncode != 0:
            raise RuntimeError(f"value_bench failed: {stderr.decode()[-2000:]}")
        return json.loads(stdout.decode())
    # Feed every process concurrently; sequential communicate() leaves other
    # copies waiting on stdin and would measure an isolated pass at every load.
    with ThreadPoolExecutor(max_workers=copies) as executor:
        return list(executor.map(finish, procs))


def worst(rows: list[dict], key: str) -> dict:
    return {q: max(r[key][q] for r in rows) for q in ("mean", "p50", "p95", "p99", "max")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", action="append", required=True, help="model-set directory (with MANIFEST.json and headers)")
    ap.add_argument("--model", default="general")
    ap.add_argument("--streams", nargs="*", default=[str(p) for p in sorted(FIXTURES.glob("stream_n*.txt.gz"))])
    ap.add_argument("--calls", type=int, default=7)
    ap.add_argument("--copies", type=int, nargs="+", default=[1])
    ap.add_argument("--verify", action="store_true", help="bit-exactness of the incremental path only; no timing")
    ap.add_argument("--out", help="output path without extension (.json, .md); required unless --verify")
    a = ap.parse_args(argv)
    streams = [gzip.decompress(Path(s).read_bytes()) if s.endswith(".gz") else Path(s).read_bytes() for s in a.streams]
    report = {"host": platform.node(), "hardware": hostinfo.profile(), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "calls_per_turn": a.calls, "streams": [Path(s).name for s in a.streams], "sets": {}}
    for d in a.set:
        header = Path(d) / f"weights_{a.model}.hpp"
        manifest = json.loads((Path(d) / "MANIFEST.json").read_text(encoding="utf-8"))
        shared = manifest[a.model].get("shared_value", {})
        verify_chunks(Path(d), shared.get("chunks", []))
        binary = native.model_tool("value_bench", header, f"fusion::models::{a.model}")
        if a.verify:
            for s in streams:
                r = subprocess.run([str(binary), "--verify", str(a.calls)], input=s, capture_output=True)
                print(f"[value_cost] {d}: {r.stdout.decode().strip()}")
                if r.returncode:
                    raise SystemExit(f"{d}: incremental and full value passes differ")
            continue
        fams = native.model_families(header)
        entry = {"families": fams, "header_bytes": header.stat().st_size, "copies": {},
                 "manifest_sha256": sha256(Path(d) / "MANIFEST.json"), "binary_sha256": sha256(binary),
                 "model_files": {p.relative_to(Path(d).resolve()).as_posix(): sha256(p)
                                 for p in native.model_dependencies(header) if p.is_relative_to(Path(d).resolve())}}
        for n in a.copies:
            rows = run_copies(binary, streams, a.calls, n)
            entry["copies"][str(n)] = {k: worst(rows, k) for k in ("view_us", "encode_us", "full_us", "incremental_us", "first_layer_us",
                                                                   "turn_full_ms", "turn_incremental_ms", "changed_inputs",
                                                                   "policy_view_us", "policy_encode_us", "policy_joint_us")}
            entry |= {k: rows[0][k] for k in ("value_in", "first_layer", "layers", "weight_bytes", "max_width")}
            entry["copies"][str(n)]["first_layer_share"] = max(r["first_layer_share"] for r in rows)
            print(f"[value_cost] {d} x{n}: turn p99 {entry['copies'][str(n)]['turn_full_ms']['p99']:.3f} ms "
                  f"(full pass p99 {entry['copies'][str(n)]['full_us']['p99']:.1f} us)", flush=True)
        report["sets"][str(d)] = entry
    if a.verify:
        return
    if not a.out:
        raise SystemExit("--out is required for timing")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    lines = [f"# Value cost on {report['host']}", "", f"{a.calls} value calls per turn; worst copy's percentiles. V part only: "
             "full-reply timing comes from the timing screen. Development-machine numbers are not evidence.", "",
             "| Set | Families | First layer | Copies | Encode p99 us | Full p99 us | Incr. p99 us | First-layer share | Turn p99 ms | Turn max ms |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d, e in report["sets"].items():
        for n, c in e["copies"].items():
            lines.append(f"| {Path(d).name} | {'/'.join(e['families'])} | {e['first_layer'][0]}x{e['first_layer'][1]} | {n} | "
                         f"{c['encode_us']['p99']:.1f} | {c['full_us']['p99']:.1f} | {c['incremental_us']['p99']:.1f} | "
                         f"{c['first_layer_share']:.3f} | {c['turn_full_ms']['p99']:.3f} | {c['turn_full_ms']['max']:.3f} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
