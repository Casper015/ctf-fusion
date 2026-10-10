"""Check the committed pilot summaries against their pool and fixture plan.

No games run. This checks published aggregates, not the private raw logs or
timing telemetry. Reported zeros are never presented as independently replayed
runtime evidence. Run before using a pilot summary to expand collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from roster.registry import identity  # noqa: E402
from training.collect import schedule_registry  # noqa: E402


def check(folder: Path) -> dict:
    if not __debug__:
        raise SystemExit("Run without Python -O; evidence assertions must remain enabled")
    summary_path, live_path = folder / "pilot_summary.json", folder / "pilot_live_profiles.json"
    s, live = (json.loads(p.read_text()) for p in (summary_path, live_path))
    pool_path = REPO / s["pool"]
    pool = json.loads(pool_path.read_text())
    computed_identity = identity({k: pool[k] for k in ("schema", "revision", "sampling", "reserved_ood_lineages", "bots")})
    assert s["pool_identity"] == pool["identity"] == computed_identity
    members = {b["name"]: b for b in pool["bots"]}
    expected, components, regimes = Counter(), Counter(), Counter()
    lineage = defaultdict(Counter)
    map_counts, planned = {}, 0
    for n in s["teams"]:
        fixtures = schedule_registry(n, s["matches_per_count"], 0, s["seed"], pool, members)
        map_counts[str(n)] = len({f.map_seed for f in fixtures})
        assert map_counts[str(n)] == s["matches_per_count"]
        assert len({f.flag_seed for f in fixtures}) == s["matches_per_count"]
        for f in fixtures:
            planned += 1
            regimes[f.regime] += 1
            components.update(f.components)
            for name in f.seats:
                b = members[name]
                assert b["lineage"] not in pool["reserved_ood_lineages"] and n in b["counts"]
                expected[b["id"], n] += 1
                lineage[n][b["lineage"]] += 1
    assert planned == s["total_matches"] == live["matches"]
    assert live["skipped_files"] == 0
    assert dict(components) == s["components"] and dict(regimes) == s["regimes"]
    assert sum(expected.values()) == s["total_seat_games"]
    assert {bid for bid, _ in expected} == set(live["bots"])

    observed, totals, drops, follow = Counter(), Counter(), Counter(), Counter()
    drop_by_count = Counter()
    mismatches = 0
    for bid, b in live["bots"].items():
        for n, metrics in b["per_count"].items():
            n = int(n)
            observed[bid, n] = metrics["kills_per_game"]["denominator"]
            for key, metric in (("kills", "kills_per_game"), ("deaths", "deaths_per_game"),
                                ("carrier_kills", "carrier_kills_per_game"), ("flag_captures", "captures_per_game"),
                                ("voluntary_drops", "voluntary_drop_rate")):
                totals[key] += metrics[metric]["numerator"]
            mismatches += metrics["_engine_mismatch_games"]
            drops[bid] += metrics["voluntary_drop_rate"]["numerator"]
            follow[bid] += metrics["drop_then_hit_within_3"]["numerator"]
            drop_by_count[str(n)] += metrics["voluntary_drop_rate"]["numerator"]
    assert observed == expected
    assert mismatches == s["engine_mismatch_games"] == 0
    for key, value in s["aggregate_combat_events"].items():
        # The summary reports one decimal; profiles contain rounded per-seat
        # fractional kill credits. Check displayed precision, not false exactness.
        assert round(totals[key], 1) == value, (key, totals[key], value)
    assert {bid for bid, value in drops.items() if value > 0} == set(s["voluntary_drops_by_bot"])
    for bid, row in s["voluntary_drops_by_bot"].items():
        assert drops[bid] == row["drops"] and follow[bid] == row["friendly_hit_3turns"]
    assert s["protocol_faults"] == s["late_replies"] == 0  # producer-reported fields, not independently replayed
    missing = [b["id"] for b in pool["bots"] if b["id"] not in live["bots"]]
    judge_seats = sum(v for (bid, _), v in observed.items() if bid.startswith("ctf-arena/Player7/judge/ctf/bots/"))
    input_hashes = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (summary_path, live_path, pool_path)}
    return {"scope": "Published aggregate consistency and deterministic fixture-plan reproduction only",
            "aggregate_checks_passed": True, "raw_logs_replayed": False, "raw_timing_rechecked": False,
            "input_hashes": input_hashes, "matches": planned, "seat_games": sum(observed.values()),
            "profiled_programs": len(live["bots"]), "planned_unique_maps": map_counts,
            "components": dict(components), "judge_seats": judge_seats,
            "profile_aggregate_events": dict(totals), "reported_faults": s["protocol_faults"],
            "reported_late_replies": s["late_replies"], "profile_engine_mismatch_count": mismatches,
            "drops_by_count": dict(drop_by_count), "not_observed_in_pilot": missing,
            "lineage_seat_counts": {str(n): dict(c) for n, c in lineage.items()},
            "reserved_lineage_appearances": {l: sum(c[l] for c in lineage.values()) for l in pool["reserved_ood_lineages"]}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence-dir", default="roster/report/training_selection_20261009")
    ap.add_argument("--out", help="optional JSON audit output; no raw timing certificate is generated")
    a = ap.parse_args(argv)
    result = check(REPO / a.evidence_dir)
    text = json.dumps(result, indent=1, allow_nan=False)
    if a.out:
        path = REPO / a.out
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
