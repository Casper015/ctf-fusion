"""Build the measured-evidence catalog for the roster selector (schema ctf-roster-evidence-1).

    python -m roster.catalog --assessments data/roster/bot_assessments.json \\
        --distance data/roster/behavior_distance.json --strength data/roster/strength.json \\
        --review roster_review.json --out data/roster/roster_evidence.json
    python docs/handoffs/diverse15_distance_tiered_policy/tools/select_roster.py \\
        --input data/roster/roster_evidence.json --out data/roster/selected15.json

Every number comes from the measured files: strength (maps per team count,
the lower confidence bound of the equal-count utility, clipped to [0, 1]),
behavior clusters and the overall noise-corrected excess matrix (clipped at
zero for the selector). What a measurement
cannot decide comes from a human review file, never from this tool:

    {"bots": {"<bot id>": {"eligible": true, "skills": ["carrier interception"],
                           "unique_skill_evidence": "probe tags and screen references",
                           "lineage": "optional override of the code lineage"}}}

A bot without a review entry is listed as not eligible, so the selector
reports why instead of guessing. Bots without fingerprints on every team count
have no complete distance and are left out with a reason. The manifest hash
covers every input file, so a selection is traceable to its evidence.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from fusion.paths import REPO, TEAM_COUNTS
from roster.common import load_json, save_json, sha256_bytes, sha256_file


def build(assessments: dict, dist: dict, strength: dict, review: dict, manifest_sha: str) -> dict:
    ids = dist["ids"]
    index = {b: i for i, b in enumerate(ids)}
    # the corrected (noise-adjusted) excess, clipped at zero for the selector; schema-1 files only have raw "overall"
    overall = np.array([[np.nan if v is None else max(0.0, v) for v in row] for row in dist.get("overall_excess", dist["overall"])],
                       float)
    rows = {r["id"]: r for r in assessments["bots"]}
    bots, left_out = [], {}
    for bid in ids:
        r = rows.get(bid)
        s = strength["candidates"].get(bid)
        if r is None or r["excluded"]:
            left_out[bid] = "not an eligible inventory representative"
            continue
        if s is None:
            left_out[bid] = "no strength screen"
            continue
        rv = review.get("bots", {}).get(bid, {})
        per = {str(n): s["per_count"].get(str(n), {}).get("maps", 0) for n in TEAM_COUNTS}
        lcb = s["lcb"]
        bots.append({"id": bid, "source_sha256": r["source_sha256"], "lineage": rv.get("lineage") or r["lineage_group"],
                     "behavior_cluster": dist["cluster_of"][bid], "eligible": bool(rv.get("eligible", False)),
                     "independent_maps": sum(per.values()), "maps_by_count": per,
                     "quality_lcb": 0.0 if lcb is None or not math.isfinite(lcb) else min(1.0, max(0.0, lcb)),
                     "skills": rv.get("skills", []), "unique_skill_evidence": rv.get("unique_skill_evidence", ""),
                     "evidence": {"strength": s["evidence"], "utility": s["utility"], "distinctive_tags": dist["distinctive_tags"].get(bid),
                                  "aa_noise": dist["noise"].get(bid), "reviewed": bid in review.get("bots", {})}})
    while bots:  # drop the bot with the most missing distances until the matrix is complete
        keep = [index[b["id"]] for b in bots]
        missing = (~np.isfinite(overall[np.ix_(keep, keep)])).sum(1)
        if not missing.any():
            break
        worst = int(np.argmax(missing))
        left_out[bots[worst]["id"]] = "behavior distance missing against some candidates (incomplete fingerprints)"
        bots.pop(worst)
    keep = [index[b["id"]] for b in bots]
    sub = overall[np.ix_(keep, keep)]
    sub = np.clip((sub + sub.T) / 2, 0.0, 1.0)
    np.fill_diagonal(sub, 0.0)
    return {"schema": "ctf-roster-evidence-1", "evidence_manifest_sha256": manifest_sha, "bots": bots,
            "distance_ids": [b["id"] for b in bots], "distances": [[round(float(x), 6) for x in row] for row in sub],
            "left_out": left_out}


def review_template(assessments: dict) -> dict:
    """A review file to fill in: every evaluable representative, not eligible until a person says so.
    The hints are measurements to check, not verified skills."""
    bots = {}
    for r in assessments["bots"]:
        if r["excluded"] or r["evidence_status"] in ("alias", "excluded"):
            continue
        bots[r["id"]] = {"eligible": False, "skills": [], "unique_skill_evidence": "", "lineage": None,
                         "hints": {"evidence_status": r["evidence_status"], "lineage_group": r.get("lineage_group"),
                                   "behavior_cluster": r.get("behavior_cluster"), "distinctive_tags": r.get("distinctive_tags"),
                                   "behavior_nearest": r.get("behavior_nearest"), "behavior_distance": r.get("behavior_distance"),
                                   "utility_lcb": r.get("utility_lcb"), "maps": r.get("maps"),
                                   "valid_reply_share": r.get("valid_reply_share"), "faults_per_game": r.get("faults_per_game")}}
    return {"_comment": "Set eligible true, list verified skills, and write one sentence on what this bot contributes that "
                        "the others do not, citing probe tags or reports. hints are measurements, not verdicts.", "bots": bots}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assessments", required=True)
    ap.add_argument("--distance")
    ap.add_argument("--strength")
    ap.add_argument("--review", help="human review JSON (eligibility, verified skills, contribution)")
    ap.add_argument("--review-template", help="write a review file to fill in, from the assessments, and stop")
    ap.add_argument("--out", default=str(REPO / "data" / "roster" / "roster_evidence.json"))
    a = ap.parse_args(argv)
    if a.review_template:
        save_json(a.review_template, review_template(load_json(a.assessments)))
        print(f"[catalog] review template -> {a.review_template}")
        return
    if not (a.distance and a.strength and a.review):
        ap.error("--distance, --strength and --review are required to build the catalog")
    files = {"assessments": a.assessments, "distance": a.distance, "strength": a.strength, "review": a.review}
    manifest = {k: {"path": str(Path(v).resolve()), "sha256": sha256_file(v)} for k, v in files.items()}
    manifest_sha = sha256_bytes(repr(sorted((k, v["sha256"]) for k, v in manifest.items())).encode())
    catalog = build(load_json(a.assessments), load_json(a.distance), load_json(a.strength), load_json(a.review), manifest_sha)
    catalog["manifest"] = manifest
    save_json(a.out, catalog)
    print(f"[catalog] {len(catalog['bots'])} candidates ({sum(b['eligible'] for b in catalog['bots'])} reviewed eligible), "
          f"{len(catalog['left_out'])} left out -> {a.out}")


if __name__ == "__main__":
    main()
