"""Print the loaded-vs-isolated capacity comparison (the numbers behind MULTITHREADING_ANALYSIS.md).

    python tools/analyze_multithreading.py [--multi reports/capacity_HOST.json] [--single reports/capacity_1seat_HOST.json]

Same inputs and computations as tools/generate_multithreading_report.py; this
prints them instead of writing the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_multithreading_report import compute, default_paths, ranges  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--multi")
    ap.add_argument("--single")
    ap.add_argument("--cpu")
    a = ap.parse_args(argv)
    multi, single = default_paths(a.multi, a.single)
    r = compute(json.loads(multi.read_text(encoding="utf-8")), json.loads(single.read_text(encoding="utf-8")), a.cpu)
    th = r["throughput"]
    print(f"Host {r['host']} ({r['cpu']}); {r['seats_multi']} seats at {r['time_multi']}, {r['seats_single']} at {r['time_single']}")
    print(f"Throughput: {th['single']:.0f} -> {th['multi']:.0f} seat-turns/s ({r['scaling']:.2f}x, "
          f"{100 * r['scaling'] / r['seats_multi']:.1f}% of linear)")
    for t in r["tiers"]:
        p = r["per_tier"][t]
        lo, mean, hi = p["need_ratio"]
        print(f"Tier {t:4s}: needed-time slowdown {mean:.2f}x ({lo:.2f}-{hi:.2f}), policy {p['policy_ratio']:.2f}x, "
              f"value {p['value_ratio']:.2f}x; fits {ranges(p['fits_single'])} alone, {ranges(p['fits_multi'])} loaded")
    cm = r["cost"]["multi"]
    print(f"{r['teams_cost']} teams loaded: policy = {cm['fit_fixed_ms']:.3f} ms + {cm['fit_ns_per_mac']:.3f} ns per multiply-add")
    print(f"Ceiling: {r['ceiling_multi']} loaded, {r['ceiling_single']} alone")


if __name__ == "__main__":
    main()
