"""How big can the networks be in a real competition? A capacity benchmark for the target host.

    python tools/capacity_bench.py --ai-test C:/path/to/AI_Test            # team counts 2-15, tiers S M L XL
    python tools/capacity_bench.py --raw data/raw --teams 2 6 10 15        # replay collected matches instead
    python tools/capacity_bench.py --ai-test ... --tiers S M L XL XXL --tier "W squad=192,64 value=384,32"
    python tools/capacity_bench.py --verify                                 # correctness only, no timing

Run the timing on the machine that will host the bot (or one like the judge's),
with nothing else heavy running. Development machines only run --verify.

What a real competition turn demands, and how the benchmark reproduces it:
  * Every team count 2-15 on its own map (23 / 31 / 41 / 49 cells): one recorded
    match per count, from --raw (real bots) or played through the engine with
    instant test bots (--ai-test). Unit counts, candidates and contact come from
    real states.
  * Every seat busy at once: the engine runs all bots of a match together, so
    --seats copies (default: the team count) run concurrently. At 15 teams the
    benchmarks showed this load alone makes even trivial bots time out.
  * The bot's inference per turn: the policy for every opponent team with every
    head (an upper bound), plus --value-calls value-network search nodes
    (default 24, one per root plan).
  * The worst turns: p99 and max over every turn of every copy, not means.
  * A slower judge machine: every time is multiplied by --safety (default 1.5).

Kernels: every run first checks that the SIMD multiply-add kernels give the
original scalar loop's bits on this CPU (tools/kernel_check.cpp) and stops if
not. With --kernels both (the default) every tier is timed twice, with the
SIMD kernels (avx2 / sse2 / neon, chosen at startup) and with the scalar loop
(-DFUSION_SIMD=0), and the report shows the speedup. Verdicts use the SIMD
kernels, which is what the bot runs.

The budget is what the turn leaves for inference: compute_target - search from
profiles/ (35 - 30 = 5 ms), or --budget-ms. A tier fits a team count when
safety x (policy p99 + value p99 per turn) <= budget. The report also gives
how many value nodes fit after the policy, the source size of each tier's
header (per model and for all 11 exported models: the judge may limit source
size), and the FUSION_MAX_WIDTH the tier needs. The largest tier that fits
every team count is the recommended ceiling for training.

Output: reports/capacity_<host>.md and .json.
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from fusion import hostinfo, matchlog, native, quant, synthetic  # noqa: E402
from fusion.models import PolicyNet, ValueNet  # noqa: E402
from fusion.paths import TEAM_COUNTS, load_profile  # noqa: E402
from fusion.schema import layout as feature_layout  # noqa: E402

# Hidden widths per network. S is today's default architecture; each step roughly doubles the cost.
TIERS = {
    "S": {"mover": [16], "squad": [64, 32], "act": [16], "target": [8], "value": [128, 16]},
    "M": {"mover": [32], "squad": [128, 64], "act": [32], "target": [16], "value": [256, 32]},
    "L": {"mover": [64], "squad": [256, 64], "act": [64], "target": [16], "value": [512, 64]},
    "XL": {"mover": [128], "squad": [512, 128], "act": [128], "target": [32], "value": [1024, 64]},
    "XXL": {"mover": [256], "squad": [1024, 256], "act": [256], "target": [32], "value": [2048, 128]},
}
EXPORTED_MODELS = 11  # 10 dedicated + general (training/specs/MODEL_MATRIX.md)
NETS = ("mover", "squad", "act", "target", "value")


def parse_tier(text: str) -> tuple[str, dict]:
    """'NAME squad=192,64 value=384,32': unspecified networks keep the S widths."""
    name, *parts = text.split()
    widths = {k: list(v) for k, v in TIERS["S"].items()}
    for kv in parts:
        key, val = kv.split("=")
        if key not in NETS:
            raise SystemExit(f"--tier {text!r}: unknown network {key!r} (known: {', '.join(NETS)})")
        widths[key] = [int(w) for w in val.split(",")]
    return name, widths


def tier_line(name: str, widths: dict) -> str:
    return name + " " + " ".join(f"{k}={','.join(map(str, widths[k]))}" for k in NETS)


def tier_static(widths: dict, layout) -> dict:
    """Size facts that need no timing: parameters, MACs, header bytes, the FUSION_MAX_WIDTH it needs."""
    from training import export_weights  # noqa: PLC0415
    rng = np.random.default_rng(0)
    policy = PolicyNet(layout, rng, {k: widths[k] for k in ("mover", "squad", "act", "target")})
    value = ValueNet(layout, rng, widths["value"])
    nets = export_weights.quantize_model(policy, value, 1.0, layout, max_width=1 << 16)
    text = export_weights.header("probe", nets, {}, {})
    wide = export_weights.widest(nets)
    return {"params": {k: int(sum(l["w"].size + l["b"].size for l in q["layers"])) for k, q in nets.items()},
            "macs_per_opponent_team": 15 * export_weights.macs(nets["mover"]) + export_weights.macs(nets["squad"]),
            "value_macs": export_weights.macs(nets["value"]), "header_kib": len(text.encode()) / 1024,
            "all_models_mib": EXPORTED_MODELS * len(text.encode()) / 2 ** 20, "widest_layer": wide,
            "max_width": max(quant.MAX_WIDTH, 64 * math.ceil(wide / 64))}


def plan(per_seat: list[dict], budget_ms: float, safety: float) -> dict:
    """Combine the size_bench lines of every concurrent seat for one tier and team count into a verdict."""
    worst = lambda path: max(r[path[0]][path[1]] for r in per_seat)
    policy_p99, policy_max = worst(("policy_turn_ms", "p99")), worst(("policy_turn_ms", "max"))
    value_p99 = worst(("value_turn_ms", "p99"))
    node_us = float(np.mean([r["value_node_us"]["encode"] + r["value_node_us"]["incremental"] for r in per_seat]))
    need = safety * (policy_p99 + value_p99)
    spare_ms = budget_ms / safety - policy_p99
    return {"policy_p99_ms": policy_p99, "policy_max_ms": policy_max, "policy_joint_p99_ms": worst(("policy_joint_ms", "p99")),
            "value_turn_p99_ms": value_p99, "value_node_us": node_us, "value_full_us": worst(("value_node_us", "full")),
            "need_ms": need, "fits": need <= budget_ms, "policy_fits": safety * policy_p99 <= budget_ms,
            "value_nodes_affordable": int(max(0.0, spare_ms) * 1000 / node_us) if node_us > 0 else 0,
            "seats": len(per_seat), "turns": sum(r["turns"] for r in per_seat)}


def recorded_stream(teams: int, raw: Path | None, ai_test: str | None, work: Path) -> tuple[str, str]:
    """(seat stream, source) of one real-sized match: a collected one if --raw has it, else a test-bot match."""
    if raw:
        for path in matchlog.iter_matches(raw, [teams]):
            return matchlog.seat_stream(matchlog.load(path), 0), f"{path.name} (collected)"
    if not ai_test:
        raise SystemExit(f"no recorded {teams}-team match: pass --raw with collected matches or --ai-test to play one")
    from training import collect  # noqa: PLC0415
    out = work / "raw"
    collect.main(["--ai-test", ai_test, "--teams", str(teams), "--matches", "1", "--bots", "Random", "--out", str(out)])
    path = next(iter(matchlog.iter_matches(out, [teams])))
    return matchlog.seat_stream(matchlog.load(path), 0), f"{path.name} (test bots, real map and engine)"


def run_seats(binary: Path, stream_file: Path, tiers_file: Path, seats: int, repeat: int, value_calls: int) -> list[list[dict]]:
    """`seats` concurrent copies of size_bench on one stream; per copy, its JSON line per tier."""
    procs = []
    for _ in range(seats):
        with stream_file.open("rb") as f:
            procs.append(subprocess.Popen([str(binary), str(repeat), str(value_calls), str(tiers_file)], stdin=f,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    out = []
    for p in procs:
        stdout, stderr = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"size_bench failed: {stderr.decode(errors='replace')[-1000:]}")
        out.append([json.loads(line) for line in stdout.decode().splitlines() if line.strip()])
    return out


def kernel_parity() -> str:
    """Stop unless every kernel path this CPU runs matches the scalar loop bit for bit."""
    ok, summary = native.run_kernel_check()
    print(f"[capacity] {summary}", flush=True)
    if not ok:
        raise SystemExit("SIMD kernels disagree with the scalar loop on this CPU: do not use them; report this output")
    return summary


def verify(tiers: dict, teams_list: list[int]) -> int:
    """Correctness only (no timing): kernel parity, then incremental value == full pass for every tier."""
    kernel_parity()
    layout = feature_layout()
    width = max(tier_static(w, layout)["max_width"] for w in tiers.values())
    binary = native.size_bench(width)
    failures = 0
    with tempfile.TemporaryDirectory(prefix="capacity_") as tmp:
        tiers_file = Path(tmp) / "tiers.txt"
        tiers_file.write_text("\n".join(tier_line(n, w) for n, w in tiers.items()) + "\n", encoding="utf-8")
        for teams in teams_list:
            r = native.run_tool(binary, synthetic.stream(teams, turns=12).encode(), args=("--verify", 1, 24, tiers_file))
            lines = [json.loads(l) for l in r.stdout.decode().splitlines() if l.strip()]
            ok = r.returncode == 0 and len(lines) == len(tiers) and all(l["verified"] for l in lines)
            failures += not ok
            print(f"[verify] {teams} teams, FUSION_MAX_WIDTH={width}: {'ok' if ok else 'FAILED'} "
                  f"({', '.join(l['tier'] + ' ' + str(l['nodes']) + ' nodes' for l in lines)}){r.stderr.decode()[-300:]}", flush=True)
    return 1 if failures else 0


def kernel_lines(report: dict) -> list[str]:
    """Scalar loop against SIMD kernels, per team count and tier (when both were timed)."""
    if not any("scalar" in c.get("kernels", {}) for c in report["counts"]):
        return []
    f = lambda v, d=2: f"{v:.{d}f}"
    lines = ["", "## Kernels: scalar loop vs SIMD", "",
             f"SIMD path on this CPU: **{report.get('simd_path', '?')}**. Same integers, same results "
             f"({report.get('kernel_check', '')}). Speedup = scalar time / SIMD time.", "",
             "| teams | tier | P p99 scalar / SIMD ms | P speedup | V node scalar / SIMD us | V speedup | needed ×safety scalar / SIMD ms |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for c in report["counts"]:
        for name, s in c["kernels"]["scalar"].items():
            v = c["kernels"]["simd"][name]
            lines.append(f"| {c['teams']} | {name} | {f(s['policy_p99_ms'], 3)} / {f(v['policy_p99_ms'], 3)} | "
                         f"{s['policy_p99_ms'] / v['policy_p99_ms']:.2f}× | {f(s['value_node_us'])} / {f(v['value_node_us'])} | "
                         f"{s['value_node_us'] / v['value_node_us']:.2f}× | {f(s['need_ms'])} / {f(v['need_ms'])} |")
    return lines


def markdown(report: dict) -> str:
    f = lambda v, d=2: f"{v:.{d}f}"
    host = report.get("host_profile", {})
    lines = [f"# Capacity benchmark: {report['host']}", "", report["time"], "",
             f"CPU {host.get('cpu', 'not recorded')}, {host.get('logical_cpus', '?')} logical processors, {host.get('os', '')}. "
             f"SIMD kernels: {report.get('simd_path', 'not recorded')}.", "",
             f"Budget {report['budget_ms']} ms per turn for inference (compute target minus search), safety factor "
             f"{report['safety']}, {report['value_calls']} value nodes per turn, every seat of a match running at once. "
             "P is the policy for every opponent team with every head (an upper bound); V is the value network per "
             "search node. Times are the worst p99 over every turn of every concurrent seat.", "",
             "## Tiers", "", "| tier | hidden widths | P MACs / opponent team | V MACs / node | header per model | all 11 models | FUSION_MAX_WIDTH |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, t in report["tiers"].items():
        w = t["widths"]
        desc = "; ".join(f"{k} {','.join(map(str, w[k]))}" for k in NETS)
        s = t["static"]
        lines.append(f"| {name} | {desc} | {s['macs_per_opponent_team']:,} | {s['value_macs']:,} | {f(s['header_kib'], 0)} KiB | "
                     f"{f(s['all_models_mib'], 1)} MiB | {s['max_width']} |")
    lines += ["", "## Per team count", "", "| teams | map | seats | tier | P p99 / max ms | V per node us | V per turn p99 ms | "
              f"needed ×{report['safety']} ms | fits | V nodes that fit |", "|---:|---:|---:|---|---:|---:|---:|---:|---|---:|"]
    for c in report["counts"]:
        for name, p in c["tiers"].items():
            lines.append(f"| {c['teams']} | {c['map_size']} | {p['seats']} | {name} | {f(p['policy_p99_ms'], 3)} / {f(p['policy_max_ms'], 3)} | "
                         f"{f(p['value_node_us'])} | {f(p['value_turn_p99_ms'], 3)} | {f(p['need_ms'])} | "
                         f"{'yes' if p['fits'] else 'NO'} | {p['value_nodes_affordable']:,} |")
    lines += ["", "## Verdict", ""]
    for c in report["counts"]:
        lines.append(f"- {c['teams']} teams: largest tier that fits: **{c['largest_fit'] or 'none'}**")
    lines += ["", f"**Recommended ceiling for every team count: {report['ceiling'] or 'none of the tiers'}.**", "",
              "Read with care:", "",
              "- The policy figure includes every head at every legal move; the joint-v2 adapter needs the heads only at "
              "sampled moves, so real cost is lower. Search time is not included: the budget is what search leaves.",
              "- Larger networks only help if training data supports them; this table bounds cost, not benefit.",
              "- A tier above FUSION_MAX_WIDTH 256 must be trained and exported with `--max-width` and the bot built with "
              "`-DFUSION_MAX_WIDTH`; the exported header refuses to compile otherwise.",
              "- The judge's source-size limit is unknown; the last tier columns give the header text it would add.", "",
              "## Streams", "", *[f"- {c['teams']} teams: {c['source']}" for c in report["counts"]]]
    lines += kernel_lines(report)
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ai-test", help="AI_Test checkout: plays one test-bot match per team count for its states")
    ap.add_argument("--raw", help="collected matches (data/raw) to replay instead")
    ap.add_argument("--teams", type=int, nargs="+", default=list(TEAM_COUNTS))
    ap.add_argument("--tiers", nargs="+", default=["S", "M", "L", "XL"], help=f"named tiers: {', '.join(TIERS)}")
    ap.add_argument("--tier", action="append", default=[], help="extra tier: 'NAME squad=192,64 value=384,32 ...'")
    ap.add_argument("--seats", type=int, default=0, help="concurrent copies per match (default: the team count)")
    ap.add_argument("--repeat", type=int, default=2, help="passes over the recorded turns per copy")
    ap.add_argument("--value-calls", type=int, default=24, help="value-network search nodes per turn")
    ap.add_argument("--budget-ms", type=float, default=0.0, help="inference budget per turn (default: profile compute_target - search)")
    ap.add_argument("--safety", type=float, default=1.5, help="multiplier for a slower judge machine")
    ap.add_argument("--kernels", choices=("both", "simd", "scalar"), default="both",
                    help="time the SIMD kernels, the scalar loop, or both (verdicts use the first of simd, scalar)")
    ap.add_argument("--verify", action="store_true", help="correctness only, no timing (any machine)")
    ap.add_argument("--tag", help="name the report capacity_<tag>_<host> (e.g. --seats 1 --tag 1seat for the isolated run)")
    ap.add_argument("--out", default=str(REPO / "reports"))
    a = ap.parse_args(argv)
    tiers = {n: TIERS[n] for n in a.tiers} | dict(parse_tier(t) for t in a.tier)
    if a.verify:
        return verify(tiers, a.teams)
    parity = kernel_parity()
    layout = feature_layout()
    statics = {n: tier_static(w, layout) for n, w in tiers.items()}
    width = max(s["max_width"] for s in statics.values())
    kernels = ["simd", "scalar"] if a.kernels == "both" else [a.kernels]
    binaries = {k: native.size_bench(width, simd=k == "simd") for k in kernels}
    report = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "safety": a.safety,
              "value_calls": a.value_calls, "max_width_built": width, "counts": [], "host_profile": hostinfo.profile(),
              "kernels": kernels, "kernel_check": parity, "seats_arg": a.seats or "team count",
              "tiers": {n: {"widths": tiers[n], "static": statics[n]} for n in tiers}}
    with tempfile.TemporaryDirectory(prefix="capacity_") as tmp:
        work = Path(tmp)
        tiers_file = work / "tiers.txt"
        tiers_file.write_text("\n".join(tier_line(n, w) for n, w in tiers.items()) + "\n", encoding="utf-8")
        for teams in a.teams:
            profile = load_profile(teams)
            budget = a.budget_ms or profile["turn_budget_ms"]["compute_target"] - profile["turn_budget_ms"]["search"]
            report.setdefault("budget_ms", budget)
            stream, source = recorded_stream(teams, Path(a.raw) if a.raw else None, a.ai_test, work)
            stream_file = work / f"n{teams:02d}.txt"
            stream_file.write_bytes(stream.encode())
            seats = a.seats or teams
            entry = {"teams": teams, "source": source, "budget_ms": budget, "kernels": {}, "seconds_by_kernel": {}}
            for k in kernels:
                t0 = time.time()
                copies = run_seats(binaries[k], stream_file, tiers_file, seats, a.repeat, a.value_calls)
                entry["seconds_by_kernel"][k] = round(time.time() - t0, 1)
                entry["kernels"][k] = {name: plan([c[i] for c in copies], budget, a.safety) for i, name in enumerate(tiers)}
                entry["map_size"] = copies[0][0]["map_size"]
                if k == "simd":
                    report["simd_path"] = copies[0][0]["kernel"]
            entry["tiers"] = entry["kernels"][kernels[0]]
            entry["seconds"] = entry["seconds_by_kernel"][kernels[0]]
            fitting = [n for n in tiers if entry["tiers"][n]["fits"]]
            entry["largest_fit"] = max(fitting, key=lambda n: statics[n]["macs_per_opponent_team"]) if fitting else None
            report["counts"].append(entry)
            print(f"[capacity] {teams} teams ({seats} seats, {entry['seconds']} s, {kernels[0]}): "
                  + ", ".join(f"{n} {p['need_ms']:.2f} ms{'' if p['fits'] else ' (over)'}" for n, p in entry["tiers"].items())
                  + ("" if len(kernels) < 2 else "; P speedup over scalar " + ", ".join(
                      f"{n} {entry['kernels']['scalar'][n]['policy_p99_ms'] / entry['kernels']['simd'][n]['policy_p99_ms']:.1f}x"
                      for n in tiers)), flush=True)
    everywhere = [n for n in tiers if all(c["tiers"][n]["fits"] for c in report["counts"])]
    report["ceiling"] = max(everywhere, key=lambda n: statics[n]["macs_per_opponent_team"]) if everywhere else None
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = out / (f"capacity_{a.tag}_{report['host']}" if a.tag else f"capacity_{report['host']}")
    stem.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    stem.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(f"[capacity] wrote {stem.with_suffix('.md')}; ceiling: {report['ceiling']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
