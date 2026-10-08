"""One assessment row per candidate: identity, build, aliases, lineage, reliability, behavior, strength, evidence.

    python -m roster.assess --inventory data/roster/bot_inventory.json --fingerprints data/roster/fingerprints \\
        --distance data/roster/behavior_distance.json [--strength data/roster/strength.json] --out data/roster/assessments

Writes all_bot_assessments.csv (flat, one row per inventoried candidate,
excluded ones included with their reason) and bot_assessments.json (full).
A byte alias carries its representative's behavior and strength, marked as
inherited; it never counts as independent evidence.

Evidence status, lowest to highest: excluded, alias, inventoried (built),
fingerprinted (probe replies), screened (strength screen), confirmed (at least
10,000 maps with 500 per team count). Static inspection and fingerprints are
mechanism evidence; only screened or confirmed rows carry strength.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from fusion.paths import REPO
from roster.common import load_json, save_json
from roster.fingerprint import load_all

COLUMNS = ["id", "pool_name", "evidence_status", "excluded", "representative", "byte_alias_of", "code_alias_of",
           "lineage_group", "code_nearest", "code_similarity", "build", "binary_sha256", "source_sha256",
           "probe_runs", "valid_reply_share", "probe_faults", "probe_dead_runs", "probe_reply_ms_p50", "probe_reply_ms_p99",
           "behavior_cluster", "behavior_nearest", "behavior_distance", "aa_noise", "distinctive_tags",
           "utility", "utility_lcb", "maps", "counts_covered", "strength_evidence", "faults_per_game", "late_replies",
           "p99_ms_median", "max_ms", "first_ms_max"]


def reliability(runs: list[dict]) -> dict:
    if not runs:
        return {}
    valid = np.concatenate([np.asarray(r["valid"], bool) for r in runs])
    ms = np.array([x for r in runs for x in r["reply_ms"] if x is not None], float)
    return {"probe_runs": len(runs), "valid_reply_share": round(float(valid.mean()), 4),
            "probe_faults": sum(len(r["faults"]) for r in runs), "probe_dead_runs": sum(r["dead"] is not None for r in runs),
            "probe_reply_ms_p50": round(float(np.percentile(ms, 50)), 3) if len(ms) else None,
            "probe_reply_ms_p99": round(float(np.percentile(ms, 99)), 3) if len(ms) else None}


def assess(inv: dict, runs: dict, dist: dict | None, strength: dict | None) -> list[dict]:
    by_id = {b["id"]: b for b in inv["bots"]}
    ids = dist["ids"] if dist else []
    idx = {b: i for i, b in enumerate(ids)}
    rows = []
    for b in inv["bots"]:
        rep = b["representative"]
        row = {"id": b["id"], "pool_name": b.get("pool_name"), "excluded": b["excluded"], "representative": b["representative"],
               "byte_alias_of": b["byte_alias_of"], "code_alias_of": b.get("code_alias_of"), "lineage_group": b["lineage_group"],
               "code_nearest": (b["nearest_relative"] or {}).get("id"), "code_similarity": (b["nearest_relative"] or {}).get("similarity"),
               "build": b["build"].get("status"), "binary_sha256": b["build"].get("binary_sha256"),
               "source_sha256": b["source_sha256"], "inherited_from": None if rep == b["id"] else rep}
        row |= reliability(runs.get(rep, []))
        if dist and rep in idx:
            row |= {"behavior_cluster": dist["cluster_of"].get(rep), "aa_noise": dist["noise"].get(rep),
                    "behavior_nearest": (dist["nearest"].get(rep) or {}).get("id"),
                    "behavior_distance": (dist["nearest"].get(rep) or {}).get("distance"),
                    "distinctive_tags": ", ".join(f"{t['tag']} x{t['vs_population']}" for t in dist["distinctive_tags"].get(rep, [])),
                    "clusters_per_count": {n: next((c[0] for c in cl if rep in c), None) for n, cl in dist["clusters_per_count"].items()}}
        s = (strength or {}).get("candidates", {}).get(rep)
        if s:
            row |= {"utility": s["utility"], "utility_lcb": s["lcb"], "maps": s["maps"], "counts_covered": s["team_counts_covered"],
                    "strength_evidence": s["evidence"], "faults_per_game": s["faults_per_game"], "late_replies": s["late_replies"],
                    "p99_ms_median": s["p99_ms_median"], "max_ms": s["max_ms"], "first_ms_max": s["first_ms_max"],
                    "per_count": s["per_count"]}
        if rep != b["id"]:
            status = "alias"
        elif b["excluded"]:
            status = "excluded"
        elif s and s["evidence"] == "confirmation":
            status = "confirmed"
        elif s:
            status = "screened"
        elif row.get("probe_runs"):
            status = "fingerprinted"
        else:
            status = "inventoried"
        row["evidence_status"] = status
        rows.append(row)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--fingerprints", default=str(REPO / "data" / "roster" / "fingerprints"))
    ap.add_argument("--distance")
    ap.add_argument("--strength")
    ap.add_argument("--out", default=str(REPO / "data" / "roster"))
    a = ap.parse_args(argv)
    inv = load_json(a.inventory)
    runs = load_all(Path(a.fingerprints)) if Path(a.fingerprints).is_dir() else {}
    rows = assess(inv, runs, load_json(a.distance) if a.distance else None, load_json(a.strength) if a.strength else None)
    out = Path(a.out)
    save_json(out / "bot_assessments.json", {"schema": "ctf-bot-assessments-1", "inventory": a.inventory,
                                            "distance": a.distance, "strength": a.strength, "bots": rows})
    with (out / "all_bot_assessments.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    counts = {}
    for r in rows:
        counts[r["evidence_status"]] = counts.get(r["evidence_status"], 0) + 1
    print(f"[assess] {len(rows)} candidates {counts} -> {out / 'all_bot_assessments.csv'}")


if __name__ == "__main__":
    main()
