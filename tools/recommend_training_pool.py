"""Reproduce the evidence-based 2026-10-09 opponent selection; no bot matches.

Writes a separate admission, joined screen assessments, candidate ledger,
sampling comparison and collection-ready proposed pool. The active pool is
never replaced. Roles/caps are provisional collection priors, not proven
training improvements. Program IDs and source hashes remain authoritative.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from roster import registry  # noqa: E402

JUDGES = {"Astra", "Sol", "Sonnet", "Fable", "Opus", "DeepSeek", "Kimi", "MiMo", "Qwen", "GLM",
          "doubao-seed-2.1-turbo"}
CHALLENGERS = {"ctf-arena_Player5_v12_rollout_volley", "cand-mc1", "Control_external_CL", "cand-012",
               "R3_EXPECTIMAX_sampled", "C04_G_PLAYOUT_4ply", "bot_2"}
DROP_SOURCES = {"Fable", "C04_G_PLAYOUT_4ply", "C08_Player5_Omega_LIT", "Player5", "bot_2"}
MERGE = {"C20_R3_HABITS_adaptive": "R3_HABITS_adaptive", "C22_R3_WEIGHTS_tuned": "R3_WEIGHTS_tuned",
         "ctf-arena_Player4_v8_winning": "ctf-arena_Player4_v9_latest"}
HOLDOUT = {"lineage:AI_Arena_CTF_Player02_bot", "lineage:ctf-arena_Player4_v2"}
OWN = {"lineage:Player5", "lineage:Control_Hybrid"}
SNAPSHOT_JSON = {
    "profiles/open_training_pool.json": "54d67c1f700e611d48dda6ddb5551130b82ad710e5efb023237fc1cc1ce63ca2",
    "roster/report/screen_light/admission_screened.json": "1b642450eee7841b92018a93b5264cd10269914682838920b28c152d630de6e4",
    "roster/report/results/bot_assessments.json": "76c8375ffe6d6d96d8e9b46bf1a237914ca51308695fab43a3e3549686fbc3f1",
    "roster/new_bots/inventory.json": "fe06fb9c21d07bd6c02472b7550f8cbdce8b2c9525158fef6fa4365d1ab8bd8c",
    "roster/report/results/behavior_distance.json": "50de33bdaf2010098218e64359d023d1f8846a5c803ba8dd5547c56a3f55614d",
    "roster/report/screen_light/strength.json": "8b61ccf894bcbd0cc9c3da40e09b6e8e45b5ffc43c5a438860b786d4b985a8df",
}
SNAPSHOT_CODE = {
    "roster/registry.py": "f97c2b16849850eef1449d76a7a544cfa9da28d6fb1f7b29e39c1e1eb61d76d3",
    "roster/sampling.py": "57540d308c49f364a39fc52896362f4fdfc4635b5757537e9f17cd5113f88054",
    "roster/distance.py": "a3972cf20ceee0d8e3cd29c7f87e09182549990c9406da6c496f0914ecea637e",
    "roster/common.py": "14045f06ec0ec4a5f5c6771296cfb6335c374721830e86bdd63014be89845580",
}


def read(rel):
    data = json.loads((REPO / rel).read_text())
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    if hashlib.sha256(canonical).hexdigest() != SNAPSHOT_JSON[rel]:
        raise SystemExit(f"{rel}: evidence snapshot changed; re-review this recommendation before regenerating")
    return data


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def provenance(path):
    return {"path": path.relative_to(REPO).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def shares(pool):
    members = {b["name"]: b for b in pool["bots"]}
    rows = []
    for n, c in pool["per_count"].items():
        n = int(n)
        seats = c["inclusion"]["seats_per_game"]
        own = sum(v for k, v in seats.items() if members[k]["lineage"] in OWN) / n
        judges = sum(v for k, v in seats.items() if members[k]["id"].startswith("ctf-arena/Player7/judge/ctf/bots/")) / n
        rows.append({"teams": n, "own_two_lineages": own, "judge_bots": judges, "other": 1-own-judges,
                     "simulated_games": c["inclusion"]["games"],
                     "forced_repeat_seats_per_game": c["inclusion"]["forced_repeat_seats_per_game"]})
    rows.sort(key=lambda r: r["teams"])
    total = sum(r["teams"] for r in rows)
    return {"per_count": rows,
            "equal_count": {k: statistics.mean(r[k] for r in rows) for k in ("own_two_lineages", "judge_bots", "other")},
            "equal_games_team_row_weighted": {k: sum(r["teams"]*r[k] for r in rows)/total
                                               for k in ("own_two_lineages", "judge_bots", "other")}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=10000, help="sampler draws per count; not game simulations")
    ap.add_argument("--out", default="roster/report/training_selection_20261009")
    ap.add_argument("--pool-out", default="profiles/open_training_pool_recommended.json")
    a = ap.parse_args(argv)
    if a.games < 1:
        ap.error("--games must be positive")
    out = REPO / a.out
    pool_out = REPO / a.pool_out
    if pool_out.resolve() == (REPO / "profiles/open_training_pool.json").resolve():
        ap.error("cannot overwrite the active pool")
    for rel, expected in SNAPSHOT_CODE.items():
        data = (REPO / rel).read_bytes().replace(b"\r\n", b"\n")
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"{rel}: controlling implementation changed; re-review before regenerating")

    base = read("profiles/open_training_pool.json")
    adm = read("roster/report/screen_light/admission_screened.json")
    assessment = read("roster/report/results/bot_assessments.json")
    inv = read("roster/new_bots/inventory.json")
    dist = read("roster/report/results/behavior_distance.json")
    screen = read("roster/report/screen_light/strength.json")
    by_name = {b["name"]: b for b in base["bots"]}
    assert len(by_name) == len(base["bots"])
    assert (JUDGES | CHALLENGERS | DROP_SOURCES | set(MERGE) | set(MERGE.values())) <= set(by_name)
    by_inventory = {b["id"]: b for b in inv["bots"]}
    by_row = {b["id"]: b for b in assessment["bots"]}
    for bid, c in screen["candidates"].items():
        assert c["source_sha256"] == by_inventory[bid]["source_sha256"] == by_row[bid]["source_sha256"]
        assert c["games"] == c["maps"] == 70 and c["all_counts"]
        assert all(v["maps"] == 5 for v in c["per_count"].values())
        r = by_row[bid]
        r.update({"utility": c["utility"], "utility_lcb": c["lcb"], "maps": c["maps"],
                  "counts_covered": c["team_counts_covered"], "strength_evidence": c["evidence"],
                  "faults_per_game": c["faults_per_game"], "late_replies": c["late_replies"],
                  "p99_ms_median": c["p99_ms_median"], "max_ms": c["max_ms"], "first_ms_max": c["first_ms_max"],
                  "per_count": c["per_count"], "evidence_status": "screened"})
    assert not screen["stale_rows"]

    adm["revision"] = "review-20261009-balanced-pilot"
    adm["reserved_ood_lineages"] = sorted(set(adm.get("reserved_ood_lineages", [])) | HOLDOUT)
    adm["sampling"] = {"lambdas": {"field": .7, "style": .3, "hard": 0}, "lineage_cap": .1,
                       "augmentation": .02, "focus": 0, "focus_fraction": .5}
    adm["_comment"] = ("Proposed pilot, separate from the active pool. field means official benchmark targets plus seven "
                       "strong/rare-behavior challengers; it does not claim every official bot is strong. Every retained "
                       "policy also participates in style. Count limits are behavior-redundancy choices, not strength ceilings. "
                       "Two complete source lineages are reserved for fresh-map evaluation. Augmentation is 2% of fixtures "
                       "with one augmentation seat, not 2% of seats, and never supplies P labels. No training gain established.")
    ledger = []
    for b in base["bots"]:
        name, bid = b["name"], b["id"]
        spec = adm["bots"][bid]
        c = screen["candidates"][bid]
        why = "Reliable, measured behavior coverage; retain a small group-balanced style share."
        if name in MERGE:
            disposition, why = "merge", f"Use {MERGE[name]} as the operational representative; no extra physical copy."
        elif b["lineage"] in HOLDOUT:
            disposition, why = "holdout", "Reserve complete lineage for unseen-policy evaluation on fresh maps."
        elif name == "bot":
            disposition, why = "exclude_idle", "Observed no attacks, kills or captures; novelty alone is insufficient for primary training."
        elif "augmentation" in b["roles"]:
            disposition, why = "augmentation", "Controlled state perturbation only; no P labels."
        else:
            disposition = "retain"
            if name in JUDGES:
                why = "Distinct official benchmark prediction target; retain even when weak, without calling it a strong teacher."
            elif name in CHALLENGERS:
                why = "Limited challenger-field share: positive light-screen utility and measured behavior or rare-drop coverage."
            elif name in DROP_SOURCES:
                why = "Observed voluntary-drop events; preserve rare flag/action coverage independently of point rank."
            elif len(b["counts"]) < 14:
                why = "Retain only current count-specific enrichment; this does not prove a strength limit outside those counts."
            spec["roles"] = ["style"] + (["field"] if name in JUDGES | CHALLENGERS else [])
        spec["admit"] = disposition in ("retain", "augmentation")
        spec["contribution"] += " Selection review 2026-10-09: " + why
        live = c["live"]["per_count"]
        drops = sum(v["voluntary_drop_rate"]["numerator"] for v in live.values())
        follow_hits = sum(v["drop_then_hit_within_3"]["numerator"] for v in live.values())
        ledger.append({"original_pool_name": name, "id": bid, "source_sha256": b["source_sha256"],
                       "lineage": b["lineage"], "disposition": disposition, "reason": why,
                       "counts": b["counts"], "roles": spec["roles"], "utility": c["utility"], "lcb": c["lcb"],
                       "ucb": c["utility"] + 1.96*c["se"], "screen_maps": c["maps"],
                       "voluntary_drops_candidate_games": drops, "drops_followed_by_any_own_hit_3": follow_hits})
    adm_path = out / "admission_recommended.json"
    assess_path = out / "assessments_screened.json"
    save(adm_path, adm)
    save(assess_path, assessment)
    inputs = {"admission": provenance(adm_path), "assessments": provenance(assess_path),
              "distance": provenance(REPO / "roster/report/results/behavior_distance.json"),
              "inventory": provenance(REPO / "roster/new_bots/inventory.json")}
    proposed = registry.build(adm, assessment, dist, inv, inputs=inputs, games=a.games)
    assert not proposed["not_admitted"] and not proposed["counts_without_behavior_evidence"]
    assert all(b["lineage"] not in HOLDOUT for b in proposed["bots"])
    assert all("augmentation" not in b["roles"] or not b["policy_labels"] for b in proposed["bots"])
    selected = {b["id"]: b for b in proposed["bots"]}
    for r in ledger:
        r["recommended_pool_name"] = selected[r["id"]]["name"] if r["id"] in selected else None
    save(pool_out, proposed)
    ledger.sort(key=lambda r: -r["utility"])
    save(out / "candidate_selection.json", {"source_revision": "e85ff1a91f6113e7f5ea179ed47bdf4b51c19409",
         "inputs": inputs, "screen": provenance(REPO / "roster/report/screen_light/strength.json"),
         "script": provenance(Path(__file__)), "candidates": ledger})
    with (out / "candidate_selection.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ledger[0])); w.writeheader()
        for row in ledger:
            w.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    comparison = {"measure": "Expected acting-team seat share, not playing-strength gain",
                  "current": shares(base), "proposed": shares(proposed), "proposed_pool_identity": proposed["identity"]}
    save(out / "sampling_comparison.json", comparison)
    print(json.dumps({"members": len(proposed["bots"]), "P_label_sources": sum(b["policy_labels"] for b in proposed["bots"]),
                      "warnings": proposed["warnings"], "current": comparison["current"]["equal_games_team_row_weighted"],
                      "proposed": comparison["proposed"]["equal_games_team_row_weighted"]}, indent=1))


if __name__ == "__main__":
    main()
