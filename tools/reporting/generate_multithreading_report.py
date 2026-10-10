"""Compare a loaded and an isolated capacity run; every number and verdict is computed from the two JSON files.

    python tools/reporting/generate_multithreading_report.py                       # newest reports/host/capacity_<host>.json
    python tools/reporting/generate_multithreading_report.py --multi reports/host/capacity_HOST.json \\
        --single reports/host/capacity_1seat_HOST.json --counts 2 4 8 15 --out reports/host/MULTITHREADING_ANALYSIS.md

Inputs are tools/host/capacity_bench.py reports: one run with every seat busy
(--multi) and one with a single seat (--single, by default
reports/host/capacity_1seat_<host>.json). The CPU name comes from the report's
host_profile (recorded by capacity_bench since the kernel comparison); for
older reports pass --cpu. Statements about causes the runs did not measure
(clock frequencies, cache behavior) are not made; the cost per multiply-add
shows what the data can.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def ranges(counts: list[int]) -> str:
    """[2,3,4,7] -> '2–4, 7'; [] -> 'none'."""
    if not counts:
        return "none"
    out, start = [], counts[0]
    for prev, cur in zip(counts, counts[1:] + [None]):
        if cur != prev + 1:
            out.append(f"{start}" if start == prev else f"{start}–{prev}")
            start = cur
    return ", ".join(out)


def seats(run: dict) -> int:
    return next(iter(run["counts"][0]["tiers"].values()))["seats"]


def throughput(run: dict) -> tuple[float, int, float]:
    """(seat-turns per second over every tier, seat-turns, seconds) of the run's kernel timed first."""
    turns = sum(p["turns"] for c in run["counts"] for p in c["tiers"].values())
    seconds = sum(c["seconds"] for c in run["counts"])
    return turns / seconds, turns, seconds


def compute(multi: dict, single: dict, cpu: str | None = None) -> dict:
    tiers = list(multi["tiers"])
    by_teams = {c["teams"]: c for c in single["counts"]}
    pairs = [(c, by_teams[c["teams"]]) for c in multi["counts"] if c["teams"] in by_teams]
    tm, turns_m, sec_m = throughput(multi)
    ts, turns_s, sec_s = throughput(single)
    per_tier = {}
    for t in tiers:
        need = [m["tiers"][t]["need_ms"] / s["tiers"][t]["need_ms"] for m, s in pairs]
        pol = [m["tiers"][t]["policy_p99_ms"] / s["tiers"][t]["policy_p99_ms"] for m, s in pairs]
        val = [m["tiers"][t]["value_node_us"] / s["tiers"][t]["value_node_us"] for m, s in pairs]
        per_tier[t] = {"need_ratio": (min(need), float(np.mean(need)), max(need)), "policy_ratio": float(np.mean(pol)),
                       "value_ratio": float(np.mean(val)),
                       "fits_single": [s["teams"] for _, s in pairs if s["tiers"][t]["fits"]],
                       "fits_multi": [m["teams"] for m, _ in pairs if m["tiers"][t]["fits"]],
                       "need_multi": [(m["teams"], m["tiers"][t]["need_ms"]) for m, _ in pairs]}
    # Cost per multiply-add at the largest team count: the policy runs for every other team.
    big_m, big_s = pairs[-1]
    teams = big_m["teams"]
    macs = {t: multi["tiers"][t]["static"]["macs_per_opponent_team"] for t in tiers}
    vmacs = {t: multi["tiers"][t]["static"]["value_macs"] for t in tiers}
    params = {t: sum(multi["tiers"][t]["static"]["params"].values()) for t in tiers}
    cost = {}
    for label, c in (("single", big_s), ("multi", big_m)):
        x = np.array([(teams - 1) * macs[t] for t in tiers], float)
        y = np.array([c["tiers"][t]["policy_p99_ms"] for t in tiers])
        slope, fixed = np.polyfit(x, y, 1) if len(tiers) > 1 else (y[0] / x[0], 0.0)
        cost[label] = {"policy_ns_per_mac": {t: 1e6 * c["tiers"][t]["policy_p99_ms"] / ((teams - 1) * macs[t]) for t in tiers},
                       "value_ns_per_mac": {t: 1e3 * c["tiers"][t]["value_node_us"] / vmacs[t] for t in tiers},
                       "fit_fixed_ms": float(fixed), "fit_ns_per_mac": float(slope * 1e6),
                       "mac_share": {t: float(slope * (teams - 1) * macs[t] / c["tiers"][t]["policy_p99_ms"]) for t in tiers}}
    return {"host": multi["host"], "cpu": cpu or multi.get("host_profile", {}).get("cpu") or "not recorded",
            "time_multi": multi["time"], "time_single": single["time"], "budget_ms": multi.get("budget_ms"),
            "safety": multi["safety"], "value_calls": multi["value_calls"], "seats_multi": seats(multi),
            "seats_single": seats(single), "throughput": {"multi": tm, "single": ts, "turns_multi": turns_m,
                                                          "turns_single": turns_s, "seconds_multi": sec_m, "seconds_single": sec_s},
            "scaling": tm / ts, "ceiling_multi": multi.get("ceiling"), "ceiling_single": single.get("ceiling"),
            "tiers": tiers, "per_tier": per_tier, "teams_cost": teams, "cost": cost, "params": params, "macs": macs,
            "vmacs": vmacs, "widths": {t: multi["tiers"][t]["widths"] for t in tiers}, "pairs": pairs,
            "kernels": multi.get("kernels"), "simd_path": multi.get("simd_path")}


def markdown(r: dict, counts: list[int]) -> str:
    f = lambda v, d=2: f"{v:.{d}f}"
    tiers, pt = r["tiers"], r["per_tier"]
    nm, ns = r["seats_multi"], r["seats_single"]
    teams = r["teams_cost"]
    lines = [f"# Concurrency impact: {nm} seats against {ns} on {r['host']}", "",
             f"CPU: {r['cpu']}. Runs: {nm} seats at {r['time_multi']}, {ns} seat at {r['time_single']}. "
             f"Budget {r['budget_ms']} ms of inference per turn, safety ×{r['safety']}, {r['value_calls']} value nodes per turn. "
             "Generated by `tools/reporting/generate_multithreading_report.py`; every figure below is computed from the two reports.", "",
             "## Summary", ""]
    th = r["throughput"]
    lines += [f"- **Throughput**: {f(th['single'], 0)} seat-turns/s with {ns} seat, {f(th['multi'], 0)} with {nm} "
              f"(every tier counted): {r['scaling']:.2f}× ({100 * r['scaling'] / nm:.1f}% of linear).",
              "- **Slowdown per turn under load** (needed time, ×safety, loaded / isolated, mean over team counts): "
              + "; ".join(f"{t} {pt[t]['need_ratio'][1]:.2f}×" for t in tiers) + ".",
              f"- **Largest tier fitting every team count**: {r['ceiling_multi'] or 'none'} with {nm} seats, "
              f"{r['ceiling_single'] or 'none'} with {ns}."]
    for t in tiers:
        if pt[t]["fits_multi"] != pt[t]["fits_single"]:
            lo = min(v for _, v in pt[t]["need_multi"])
            hi = max(v for _, v in pt[t]["need_multi"])
            lines.append(f"- **{t}** fits {ranges(pt[t]['fits_single'])} teams with {ns} seat but "
                         f"{ranges(pt[t]['fits_multi'])} with {nm} (needed {f(lo)}–{f(hi)} ms under load).")
    lines += ["", "## By tier", "", "| tier | hidden widths | policy p99 slowdown | value node slowdown | needed-time slowdown (min–max) "
              f"| fits, {ns} seat | fits, {nm} seats |", "|---|---|---:|---:|---:|---|---|"]
    for t in tiers:
        w = r["widths"][t]
        desc = "; ".join(f"{k} {','.join(map(str, w[k]))}" for k in ("mover", "squad", "act", "target", "value"))
        lo, mean, hi = pt[t]["need_ratio"]
        lines.append(f"| {t} | {desc} | {pt[t]['policy_ratio']:.2f}× | {pt[t]['value_ratio']:.2f}× | {mean:.2f}× ({lo:.2f}–{hi:.2f}) | "
                     f"{ranges(pt[t]['fits_single'])} | {ranges(pt[t]['fits_multi'])} |")
    lines += ["", "## By team count", "", f"| teams | map | tier | policy p99 ms, {ns} / {nm} | value node us, {ns} / {nm} | "
              f"needed ×{r['safety']} ms, {ns} / {nm} | slowdown | fits under load |", "|---:|---:|---|---:|---:|---:|---:|---|"]
    for m, s in r["pairs"]:
        if counts and m["teams"] not in counts:
            continue
        for t in tiers:
            a, b = s["tiers"][t], m["tiers"][t]
            lines.append(f"| {m['teams']} | {m['map_size']} | {t} | {f(a['policy_p99_ms'], 3)} / {f(b['policy_p99_ms'], 3)} | "
                         f"{f(a['value_node_us'])} / {f(b['value_node_us'])} | {f(a['need_ms'])} / {f(b['need_ms'])} | "
                         f"{b['need_ms'] / a['need_ms']:.2f}× | {'yes' if b['fits'] else 'NO'} |")
    cs, cm = r["cost"]["single"], r["cost"]["multi"]
    lines += ["", f"## Cost per multiply-add ({teams} teams)", "",
              f"The policy runs for {teams - 1} opponent teams. Fitting policy p99 = fixed + cost × multiply-adds over the tiers: "
              f"{ns} seat {f(cs['fit_fixed_ms'], 3)} ms + {f(cs['fit_ns_per_mac'], 3)} ns per multiply-add "
              f"({1 / cs['fit_ns_per_mac']:.2f} G/s); {nm} seats {f(cm['fit_fixed_ms'], 3)} ms + {f(cm['fit_ns_per_mac'], 3)} ns "
              f"({1 / cm['fit_ns_per_mac']:.2f} G/s). The fixed part holds feature encoding, softmax and sampling setup.", "",
              f"| tier | weights per model (int16) | policy ns per multiply-add, {ns} / {nm} | value ns per multiply-add, {ns} / {nm} "
              f"| multiply-add share of policy time, {nm} seats |", "|---|---:|---:|---:|---:|"]
    for t in tiers:
        lines.append(f"| {t} | {r['params'][t] * 2 / 1024:.0f} KiB | {f(cs['policy_ns_per_mac'][t], 3)} / {f(cm['policy_ns_per_mac'][t], 3)} | "
                     f"{f(cs['value_ns_per_mac'][t], 3)} / {f(cm['value_ns_per_mac'][t], 3)} | {100 * cm['mac_share'][t]:.0f}% |")
    order = [cm["policy_ns_per_mac"][t] for t in tiers]
    rising = [f"{a}→{b}" for a, b, x, y in zip(tiers, tiers[1:], order, order[1:]) if y > x * 1.05]
    trend = ("falls or stays flat from the smallest to the largest tier: no sign of a cache or memory-bandwidth cliff in this data"
             if not rising else f"rises at {', '.join(rising)}: a sign of cache pressure at those sizes")
    lines += ["", f"Under load the policy cost per multiply-add {trend}. The weights a process holds are one model's "
              "(the table's second column), not the source text of every exported model. "
              "Clock frequencies and cache misses were not measured in these runs, so this report does not attribute "
              "the slowdown among them.", "", "## Decision inputs", "",
              f"At {teams} teams under load (needed time includes ×{r['safety']}; value nodes that still fit the budget):", "",
              "| tier | needed ms | value nodes that fit |", "|---|---:|---:|"]
    for t in tiers:
        p = r["pairs"][-1][0]["tiers"][t]
        lines.append(f"| {t} | {f(p['need_ms'])} | {p['value_nodes_affordable']:,} |")
    lines += ["", "Time decides what can run; which tier to deploy is decided by measured prediction gain per millisecond "
              "once trained models exist, with the policy and value sizes chosen separately (they compete for the same budget)."]
    if r.get("kernels") and "scalar" in r["kernels"]:
        lines += ["", f"The loaded run timed the SIMD kernels ({r['simd_path']}) and the scalar loop; see its own report's "
                  "kernel table."]
    return "\n".join(lines) + "\n"


def default_paths(multi: str | None, single: str | None) -> tuple[Path, Path]:
    reports = REPO / "reports" / "host"
    if multi is None:
        found = sorted((p for p in reports.glob("capacity_*.json") if "1seat" not in p.name), key=lambda p: p.stat().st_mtime)
        if not found:
            raise SystemExit("no reports/host/capacity_<host>.json; pass --multi")
        multi = str(found[-1])
    if single is None:
        host = json.loads(Path(multi).read_text(encoding="utf-8"))["host"]
        single = str(reports / f"capacity_1seat_{host}.json")
    return Path(multi), Path(single)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multi", help="capacity report with every seat busy")
    ap.add_argument("--single", help="capacity report with one seat (default: reports/host/capacity_1seat_<host>.json)")
    ap.add_argument("--cpu", help="CPU label for reports that predate host_profile")
    ap.add_argument("--counts", type=int, nargs="*", default=[2, 4, 8, 15], help="team counts in the per-count table (empty: all)")
    ap.add_argument("--out", default=str(REPO / "reports" / "host" / "MULTITHREADING_ANALYSIS.md"))
    a = ap.parse_args(argv)
    multi_path, single_path = default_paths(a.multi, a.single)
    if not single_path.exists():
        raise SystemExit(f"{single_path} not found; run capacity_bench with --seats 1 or pass --single")
    r = compute(json.loads(multi_path.read_text(encoding="utf-8")), json.loads(single_path.read_text(encoding="utf-8")), a.cpu)
    Path(a.out).write_text(markdown(r, a.counts), encoding="utf-8")
    print(f"wrote {a.out} from {multi_path.name} and {single_path.name}")
    return r


if __name__ == "__main__":
    main()
