"""Audit a capacity pilot's real sources, mixture, splits and continuation noise without scoring test rows.

python -m training.value_capacity_audit --cache data/vcache/designated_v3 \
    --index data/dataset_designated_index.json --matches <screen_ood_20261102/matches> \
    --branches data/branches/c08_v3xl --out reports/models/v_capacity_5xl/source_audit.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from fusion.models import ValueNet
from fusion.paths import REPO, sha256
from fusion.schema import layout
from training.value_cache import cache_identity


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "index", "matches", "branches", "out"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args(argv)
    index = json.loads(Path(a.index).read_text(encoding="utf-8"))
    identities, policies, opponents, seeds, engines, faults = {}, Counter(), Counter(), Counter(), set(), 0
    man = json.loads((REPO / "bots/p_only/report/screen_ood_20261102/manifest.json").read_text(encoding="utf-8"))
    variants = man.get("variants", man)
    names = {v.get("binary_sha256"): k for k, v in variants.items() if isinstance(v, dict)}
    maps_by_split = {}
    for entry in index["matches"]:
        name = entry["file"].replace("\\", "/").split("/")[-1]
        path = Path(a.matches) / f"n{entry['teams']:02d}" / name
        payload = path.read_bytes()
        row = json.loads(gzip.decompress(payload))
        identities[f"n{entry['teams']:02d}/{name}"] = sha256(path)
        maps_by_split.setdefault(entry["split"], set()).add(row["map_seed"])
        if seed := re.match(r"s(\d+)-", row["match_id"]):
            seeds[seed[1]] += 1
        engines.add(row["engine_sha256"])
        faults += len(row["faults"])
        for seat in row["seats"]:
            if seat.get("family") == "candidate" or seat["name"] == "@candidate":
                policies[(names.get(seat["sha256"], "unmapped binary"), seat["sha256"])] += 1
            else:
                opponents[(seat["name"], seat.get("family"), seat["sha256"])] += 1
    branch_files, branch_policies, repetitions, mismatches = {}, Counter(), Counter(), []
    branch_faults = prefix_faults = dropped = 0
    for path in sorted(Path(a.branches).rglob("*.json.gz")):
        branch_files[path.relative_to(a.branches).as_posix()] = sha256(path)
        row = json.loads(gzip.decompress(path.read_bytes()))
        if "dropped" in row:
            dropped += 1
            continue
        branch_policies[row["carrier_sha256"]] += 1
        counts = Counter()
        for branch in row["branches"]:
            counts[branch["candidate"]] += 1
            mismatches.append(branch["prefix_mismatch"])
            branch_faults += len(branch["faults"])
            prefix_faults += branch["prefix_faults"]
        repetitions.update(counts.values())
    L = layout()
    shapes = {}
    for name, hidden in (("XL", [1024, 64]), ("4XL", [8192, 512]), ("5XL", [16384, 1024])):
        model = ValueNet(L, np.random.default_rng(1), hidden, "v3")
        shapes[name] = {"dimensions": model.net.sizes, "training_parameters": sum(w.size + b.size for w, b in zip(model.net.W, model.net.b)),
                        "outcome_macs": sum(w.size for w in model.net.W[:-1]) + model.net.W[-1][0].size,
                        "activation_bytes_at_2048": 2048 * hidden[0] * 4}
    result = {"experiment": "mixed-policy exploratory, not frozen-C08 deployment confirmation", "shapes": shapes,
              "cache": cache_identity(a.cache), "dataset_index_sha256": sha256(Path(a.index)),
              "row_weighting": "source weight unchanged; count_weighting=none", "dataset_options": index["options"],
              "raw_match_sha256": identities, "raw_maps": {s: sorted(v) for s, v in maps_by_split.items()},
              "candidate_policy_mixture": [{"carrier": k[0], "binary_sha256": k[1], "matches": v} for k, v in sorted(policies.items())],
              "opponents": [{"name": k[0], "family": k[1], "sha256": k[2], "seats": v} for k, v in sorted(opponents.items())],
              "collection_seeds": dict(seeds), "engine_sha256": sorted(engines), "raw_faults": faults,
              "branches": {"sha256": branch_files, "carrier_binary_roots": dict(branch_policies), "dropped_roots": dropped,
                           "repetitions_per_candidate": dict(repetitions), "faults": branch_faults, "prefix_faults": prefix_faults,
                           "prefix_mismatch": {"min": min(mismatches, default=0), "max": max(mismatches, default=0),
                                               "mean": float(np.mean(mismatches)) if mismatches else None},
                           "rank_index": json.loads((REPO / "data/rank/c08_v3xl/index.json").read_text(encoding="utf-8")),
                           "candidate_scope": "legacy single-unit movement deviations; complete-order collection is a separate continuation step"},
              "production_manifest_sha256": sha256(REPO / "inference/models_pxxl_v3xl/MANIFEST.json"),
              "policy_manifest_sha256": sha256(REPO / "inference/models_xxl/MANIFEST.json"),
              "test_usage": "hash and map-membership audit only; test labels not scored"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("shapes", "candidate_policy_mixture", "collection_seeds", "raw_faults")}, indent=1))


if __name__ == "__main__":
    main()
