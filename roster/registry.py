"""Open training pool and frozen evaluation roster (handoff BOT_SELECTION_PLAN.md, BOT_GROUP_AND_SAMPLING_REVIEW.md).

    python -m roster.registry template --assessments data/roster/bot_assessments.json \\
        --distance data/roster/behavior_distance.json --out admission.json
    python -m roster.registry build --admission admission.json --assessments data/roster/bot_assessments.json \\
        --distance data/roster/behavior_distance.json --inventory data/roster/bot_inventory.json \\
        --out profiles/open_training_pool.json
    python -m training.collect --pool profiles/open_training_pool.json --teams ...
    python -m roster.registry freeze-eval --inventory data/roster/bot_inventory.json --members ID ... \\
        [--registry profiles/open_training_pool.json] --out profiles/frozen_eval_roster.json

There is no size cap. A bot is admitted when a person admits it with a
contribution sentence (the admission file; `template` writes one with every
measured hint and nothing admitted) and it passes the measured gates:

  - an evaluable inventory representative (a byte or code alias shares its
    representative's entry and mass, it is never a second member);
  - outside the reserved out-of-distribution lineages;
  - the source hash it was measured with;
  - protocol reliability on the probes (valid reply share, no dead runs) and,
    when screened, in the strength games (faults per game);
  - fingerprinted at every team count it is admitted for (counts without
    behavior evidence are dropped from its counts and reported).

Strength is evidence, not a gate: a weaker specialist can be admitted. Its
evidence level is "confirmed" only with the strength confirmation (10,000
maps, 500 per count); everything else is "provisional".

Each member's behavior group per team count is recomputed among the admitted
members only (complete linkage on the distance file's per-count equivalence
scores), so the sampler (roster/sampling.py) gives each group equal style mass
however many versions it holds. The pool records, per count, the groups, the
effective mixture weights, rho and the inclusion probabilities the sampler
actually produces, and it doubles as a training.collect pool (name, path,
family, source_sha256, policy_labels). Its `identity` hashes membership and
sampling, so every collected match names the exact registry revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fusion.paths import AUGMENTATION_FAMILY, REPO, TEAM_COUNTS
from roster import sampling
from roster.common import load_json, save_json, sha256_file
from roster.distance import complete_clusters
from roster.pool import pool_name

SCHEMA_REGISTRY = "ctf-open-training-pool-1"
SCHEMA_EVAL = "ctf-frozen-eval-roster-1"
ROLES = ("field", "style", "hard", "augmentation")
GATES = {"min_valid_share": 0.99, "max_faults_per_game": 0.01}


def identity(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def matrix(rows) -> np.ndarray:
    return np.array([[np.nan if v is None else v for v in row] for row in rows], float)


def counts_of(spec) -> list[int]:
    if spec in (None, "all"):
        return list(TEAM_COUNTS)
    return sorted({int(n) for n in spec})


def template(assessments: dict, dist: dict) -> dict:
    """An admission file to fill in: every evaluable representative, nothing admitted, measured hints alongside."""
    margin = dist["params"]["margin"]
    bots = {}
    for r in assessments["bots"]:
        if r["excluded"] or r["evidence_status"] in ("alias", "excluded"):
            continue
        b = r["id"]
        near = dist.get("nearest_per_count", {}).get(b, {})
        bots[b] = {"admit": False, "contribution": "", "roles": ["style"], "counts": "all", "hard_counts": [],
                   "contexts": [], "max_copies": 1,
                   "hints": {"evidence_status": r["evidence_status"], "lineage_group": r.get("lineage_group"),
                             "distinct_counts": dist.get("distinct_counts", {}).get(b),
                             "nearest_per_count": near,
                             "counts_beyond_margin_of_nearest": sorted(int(n) for n, v in near.items() if v["excess"] > margin),
                             "stochastic": b in dist.get("stochastic", []), "aa_noise": dist["noise"].get(b),
                             "distinctive_tags": r.get("distinctive_tags"), "valid_reply_share": r.get("valid_reply_share"),
                             "utility_lcb": r.get("utility_lcb"), "strength_evidence": r.get("strength_evidence"),
                             "per_count_utility": {n: round(v["utility"], 4) for n, v in (r.get("per_count") or {}).items()}}}
    return {"_comment": "Admit a bot with admit true and one sentence on the behavior, state coverage or matchup response it "
                        "contributes (cite probe tags, counts or reports). roles: style (behavior coverage; every admitted "
                        "policy), field (intended strong competition), hard (verified hard case, at hard_counts), "
                        "augmentation (random/noise: never P labels, outside the mixture). counts: 'all' or a list. "
                        "Strength is not required; hints are measurements, not verdicts. Reserve whole OOD lineages "
                        "before any collection.",
            "revision": "r1", "reserved_ood_lineages": [], "sampling": dict(sampling.DEFAULTS), "gates": dict(GATES),
            "bots": bots}


def build(admission: dict, assessments: dict, dist: dict, inventory: dict, inputs: dict | None = None,
          games: int = 20000) -> dict:
    gates = {**GATES, **admission.get("gates", {})}
    sampling.settings(admission.get("sampling"))                   # validate before doing anything
    reserved = set(admission.get("reserved_ood_lineages", []))
    rows = {r["id"]: r for r in assessments["bots"]}
    inv = {b["id"]: b for b in inventory["bots"]}
    measured = {int(n) for n in dist.get("equivalence_score_per_count", {})}
    clusters_per_count = {int(n): cl for n, cl in dist.get("clusters_per_count", {}).items()}
    members, rejected, dropped_counts = [], {}, {}
    taken: set[str] = set()
    for bid, spec in sorted(admission.get("bots", {}).items()):
        if not spec.get("admit"):
            continue
        r, b = rows.get(bid), inv.get(bid)
        roles = list(spec.get("roles") or ["style"])
        why = None
        if r is None or b is None:
            why = "not in the assessments and inventory"
        elif r["evidence_status"] == "alias" or r.get("inherited_from"):
            why = f"alias of {r.get('inherited_from') or r.get('representative')}: shares that member's entry"
        elif r["excluded"]:
            why = f"excluded: {r['excluded']}"
        elif (spec.get("lineage") or r["lineage_group"]) in reserved:
            why = "reserved out-of-distribution lineage"
        elif not spec.get("contribution", "").strip():
            why = "no contribution sentence"
        elif unknown := sorted(set(roles) - set(ROLES)):
            why = f"unknown roles {unknown}"
        elif r["source_sha256"] != b["source_sha256"]:
            why = "source changed since it was measured"
        elif (r.get("valid_reply_share") or 0.0) < gates["min_valid_share"] or (r.get("probe_dead_runs") or 0) > 0:
            why = f"probe reliability: valid share {r.get('valid_reply_share')}, dead runs {r.get('probe_dead_runs')}"
        elif r.get("faults_per_game") is not None and r["faults_per_game"] > gates["max_faults_per_game"]:
            why = f"faults per strength game {r['faults_per_game']}"
        if why:
            rejected[bid] = why
            continue
        wanted = counts_of(spec.get("counts"))
        # augmentation policies are not grouped by behavior, but still need fingerprints to be checked for reliability
        counts = [n for n in wanted if n in measured and any(bid in g for g in clusters_per_count.get(n, []))]
        if missing := sorted(set(wanted) - set(counts)):
            dropped_counts[bid] = missing
        if not counts:
            rejected[bid] = "no behavior evidence at any admitted team count"
            continue
        hard = [n for n in counts_of(spec.get("hard_counts") or counts) if n in counts] if "hard" in roles else []
        level = "confirmed" if r.get("strength_evidence") == "confirmation" else "provisional"
        name = b.get("pool_name") or pool_name(bid, taken)
        taken.add(name)
        lineage = spec.get("lineage") or r["lineage_group"]
        family = AUGMENTATION_FAMILY + lineage if "augmentation" in roles else lineage    # never P labels in training
        members.append({"name": name, "path": b["path"], "family": family, "weight": 1,
                        "source_sha256": b["source_sha256"], "policy_labels": "augmentation" not in roles,
                        "id": bid, "lineage": lineage, "roles": roles, "counts": counts,
                        "hard_counts": hard, "contexts": list(spec.get("contexts", [])), "max_copies": int(spec.get("max_copies", 1)),
                        "contribution": spec["contribution"].strip(), "evidence_level": level,
                        "evidence": {"status": r["evidence_status"], "utility": r.get("utility"), "utility_lcb": r.get("utility_lcb"),
                                     "maps": r.get("maps"), "aa_noise": dist["noise"].get(bid),
                                     "stochastic": bid in dist.get("stochastic", []),
                                     "distinct_counts": dist.get("distinct_counts", {}).get(bid)}})
    # behavior groups per count among the admitted policies only
    index = {b: i for i, b in enumerate(dist["ids"])}
    per_count = {}
    for n in TEAM_COUNTS:
        here = [m for m in members if n in m["counts"] and "augmentation" not in m["roles"]]
        if here:
            score = matrix(dist["equivalence_score_per_count"][str(n)])
            k = [index[m["id"]] for m in here]
            groups = complete_clusters([m["id"] for m in here], score[np.ix_(k, k)])
            gid = {b: f"n{n}:{g[0]}" for g in groups for b in g}
            for m in here:
                m.setdefault("groups", {})[str(n)] = gid[m["id"]]
    pool = {"schema": SCHEMA_REGISTRY, "revision": admission.get("revision", "r1"),
            "sampling": sampling.settings(admission.get("sampling")), "gates": gates,
            "reserved_ood_lineages": sorted(reserved), "bots": members}
    pool["identity"] = identity({k: pool[k] for k in ("schema", "revision", "sampling", "reserved_ood_lineages", "bots")})
    for n in TEAM_COUNTS:
        if not any(n in m["counts"] for m in members):
            continue
        mix = sampling.mixture(members, n, pool["sampling"])
        if not mix["rho"]:
            continue
        name_of = {m["name"]: m for m in members}
        groups = {}
        for m in members:
            if str(n) in m.get("groups", {}):
                groups.setdefault(m["groups"][str(n)], []).append(m["name"])
        per_count[str(n)] = {"members": len(mix["rho"]), "groups": len(groups),
                             "lineages": len({name_of[x]["lineage"] for x in mix["rho"]}),
                             "lambdas": mix["lambdas"], "rho": {k: round(v, 5) for k, v in sorted(mix["rho"].items())},
                             "group_members": groups, "inclusion": sampling.inclusion(mix, games, seed=n)}
    pool["per_count"] = per_count
    pool["warnings"] = [f"{n} teams: every game has the same composition ({c['members']} policies)"
                        for n, c in per_count.items() if c["inclusion"]["fixed_composition"]]
    pool["warnings"] += [f"{n} teams: fewer policy copies than seats, {c['inclusion']['forced_repeat_seats_per_game']} "
                         "forced repeat seats per game" for n, c in per_count.items() if c["inclusion"]["forced_repeat_seats_per_game"] > 0]
    pool["warnings"] += [f"{n} teams: no field members, its weight went to the other components"
                         for n, c in per_count.items() if "field" not in c["lambdas"]]
    pool |= {"not_admitted": rejected, "counts_without_behavior_evidence": dropped_counts,
             "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "inputs": inputs or {},
             "_comment": "Generated by roster.registry build. Membership is frozen within a run; a new revision may admit, "
                         "reweight or retire members. Bot names and ids are offline metadata, never runtime features."}
    return pool


def freeze_eval(inventory: dict, ids: list[str], registry: dict | None, name: str, seed: int, maps_per_count: int,
                teams: list[int]) -> dict:
    inv = {b["id"]: b for b in inventory["bots"]}
    in_training = {m["id"]: m for m in (registry or {}).get("bots", [])}
    reserved = set((registry or {}).get("reserved_ood_lineages", []))
    taken: set[str] = set()
    members = []
    for bid in ids:
        b = inv.get(bid)
        if b is None:
            raise SystemExit(f"{bid}: not in the inventory")
        if b.get("excluded"):
            raise SystemExit(f"{bid}: excluded from the inventory ({b['excluded']})")
        nm = b.get("pool_name") or pool_name(bid, taken)
        taken.add(nm)
        lineage = b.get("lineage_group")
        members.append({"name": nm, "id": bid, "path": b["path"], "family": lineage, "weight": 1,
                        "source_sha256": b["source_sha256"], "lineage": lineage,
                        "in_training": bid in in_training, "ood_lineage": lineage in reserved})
    roster = {"schema": SCHEMA_EVAL, "name": name, "bots": members,
              "fixtures": {"seed": seed, "maps_per_count": maps_per_count, "teams": teams,
                           "seats": "every member equally often, rotated by map; weights are fixed at 1"},
              "training_registry": (registry or {}).get("identity")}
    roster["identity"] = identity(roster)
    roster["created"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return roster


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template")
    t.add_argument("--assessments", required=True)
    t.add_argument("--distance", required=True)
    t.add_argument("--out", required=True)
    b = sub.add_parser("build")
    b.add_argument("--admission", required=True)
    b.add_argument("--assessments", required=True)
    b.add_argument("--distance", required=True)
    b.add_argument("--inventory", required=True)
    b.add_argument("--games", type=int, default=20000, help="simulated games per count for the inclusion report")
    b.add_argument("--out", default=str(REPO / "profiles" / "open_training_pool.json"))
    f = sub.add_parser("freeze-eval")
    f.add_argument("--inventory", required=True)
    f.add_argument("--members", nargs="*", default=[], help="bot ids")
    f.add_argument("--selected", help="a select_roster.py output: its selected ids")
    f.add_argument("--registry", help="the open training pool, to mark training members and OOD lineages")
    f.add_argument("--name", default="core")
    f.add_argument("--seed", type=int, default=20261009)
    f.add_argument("--maps-per-count", type=int, default=500)
    f.add_argument("--teams", type=int, nargs="+", default=list(TEAM_COUNTS))
    f.add_argument("--out", default=str(REPO / "profiles" / "frozen_eval_roster.json"))
    a = ap.parse_args(argv)
    if a.cmd == "template":
        save_json(a.out, template(load_json(a.assessments), load_json(a.distance)))
        print(f"[registry] admission template -> {a.out}")
    elif a.cmd == "build":
        files = {"admission": a.admission, "assessments": a.assessments, "distance": a.distance, "inventory": a.inventory}
        inputs = {k: {"path": str(Path(v).resolve()), "sha256": sha256_file(v)} for k, v in files.items()}
        pool = build(load_json(a.admission), load_json(a.assessments), load_json(a.distance), load_json(a.inventory),
                     inputs, a.games)
        save_json(a.out, pool)
        print(f"[registry] {len(pool['bots'])} admitted, {len(pool['not_admitted'])} not admitted, "
              f"{len(pool['per_count'])} team counts, identity {pool['identity'][:12]} -> {a.out}")
        for w in pool["warnings"]:
            print(f"[registry] warning: {w}")
    else:
        ids = list(a.members) + ([s["id"] for s in load_json(a.selected)["selected"]] if a.selected else [])
        if not ids:
            ap.error("freeze-eval needs --members or --selected")
        roster = freeze_eval(load_json(a.inventory), ids, load_json(a.registry) if a.registry else None, a.name, a.seed,
                             a.maps_per_count, a.teams)
        save_json(a.out, roster)
        print(f"[registry] frozen evaluation roster {a.name}: {len(roster['bots'])} bots, identity {roster['identity'][:12]} -> {a.out}")


if __name__ == "__main__":
    main()
