"""Heuristic diverse-roster selection from measured evidence, not raw inventory.

No generated roster certifies its upstream measurements. Multi-start greedy
selection does not prove global optimality or infeasibility.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{64}", value) is not None


def select(catalog, size=15, min_maps=10000, min_distance=0.05, max_lineage=3, min_per_count=500):
    if catalog.get("schema") != "ctf-roster-evidence-1" or not sha(catalog.get("evidence_manifest_sha256")):
        raise ValueError("a measured-evidence catalog with a manifest hash is required")
    if size < 1 or min_maps < 1 or max_lineage < 1 or min_per_count < 1 or not 0 <= min_distance <= 1:
        raise ValueError("invalid selection constraints")
    bots = catalog["bots"]
    ids = [b["id"] for b in bots]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)) or not bots:
        raise ValueError("bot ids must be nonempty and unique")
    order = catalog["distance_ids"]
    if len(order) != len(set(order)) or set(order) != set(ids):
        raise ValueError("distance matrix ids must cover the catalog exactly")
    raw = np.asarray(catalog["distances"], dtype=float)
    if raw.shape != (len(bots), len(bots)) or not np.isfinite(raw).all():
        raise ValueError("invalid distance matrix shape or values")
    if (raw < 0).any() or (raw > 1).any() or not np.allclose(raw, raw.T, atol=1e-12, rtol=0) or not np.allclose(np.diag(raw), 0, atol=1e-12):
        raise ValueError("distance matrix must be symmetric, bounded and zero-diagonal")
    lookup = {name: i for i, name in enumerate(order)}
    at = [lookup[name] for name in ids]
    distance = raw[np.ix_(at, at)]
    qualified, excluded = [], {}
    for i, bot in enumerate(bots):
        q = bot.get("quality_lcb")
        if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError(f"invalid normalized quality lower bound: {bot['id']}")
        if not sha(bot.get("source_sha256")):
            raise ValueError("source hash required")
        reasons = []
        if bot.get("eligible") is not True:
            reasons.append("upstream assessment did not qualify bot")
        maps = bot.get("independent_maps", 0)
        per = bot.get("maps_by_count", {})
        if type(maps) is not int or maps < min_maps:
            reasons.append("insufficient independent confirmation maps")
        if not isinstance(per, dict) or set(per) != {str(n) for n in range(2, 16)} or any(type(v) is not int or v < min_per_count for v in per.values()):
            reasons.append("missing or insufficient per-count confirmation")
        elif sum(per.values()) != maps:
            reasons.append("count totals disagree with independent map total")
        for field in ("lineage", "behavior_cluster", "unique_skill_evidence"):
            if not isinstance(bot.get(field), str) or not bot[field].strip():
                reasons.append(f"missing {field}")
        skills = bot.get("skills", [])
        if not isinstance(skills, list) or not skills or any(not isinstance(s, str) or not s.strip() for s in skills):
            reasons.append("missing verified skill tags")
        if reasons:
            excluded[bot["id"]] = reasons
        else:
            qualified.append(i)
    weights = catalog.get("objective_weights", {"quality": 0.5, "coverage": 0.35, "skills": 0.15})
    if set(weights) != {"quality", "coverage", "skills"} or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in weights.values()) or not math.isclose(sum(weights.values()), 1.0):
        raise ValueError("objective weights must be nonnegative and sum to one")
    result = {"schema": "ctf-selected-roster-1", "evidence_manifest_sha256": catalog["evidence_manifest_sha256"],
              "constraints": {"size": size, "min_maps": min_maps, "min_distance": min_distance,
                              "max_per_lineage": max_lineage, "min_maps_per_count": min_per_count},
              "objective_weights": weights, "excluded": excluded, "qualified": len(qualified),
              "selected": [], "status": "no_feasible_roster_found_by_greedy",
              "limits": "Uses supplied evidence; heuristic search, no optimality or infeasibility proof; manually review and confirm downstream value."}
    if len(qualified) < size:
        result["status"] = "insufficient_qualified_candidates"
        return result
    cluster_counts = Counter(bots[i]["behavior_cluster"] for i in qualified)
    population = np.array([1 / cluster_counts[bots[i]["behavior_cluster"]] for i in qualified])
    population /= population.sum()
    all_skills = set().union(*(set(bots[i]["skills"]) for i in qualified))

    def objective(chosen):
        quality = sum(bots[i]["quality_lcb"] for i in chosen)/size
        coverage = float(np.dot(population, (1-distance[np.ix_(qualified, chosen)]).max(axis=1)))
        skills = len(set().union(*(set(bots[i]["skills"]) for i in chosen)))/len(all_skills)
        return weights["quality"]*quality + weights["coverage"]*coverage + weights["skills"]*skills

    def admissible(i, chosen):
        return (all(bots[i]["behavior_cluster"] != bots[j]["behavior_cluster"] for j in chosen)
                and sum(bots[i]["lineage"] == bots[j]["lineage"] for j in chosen) < max_lineage
                and all(distance[i, j] >= min_distance for j in chosen))

    best, best_value = None, -math.inf
    for seed in qualified:
        chosen = [seed]
        while len(chosen) < size:
            choices = [i for i in qualified if i not in chosen and admissible(i, chosen)]
            if not choices:
                break
            choices.sort(key=lambda i: (-objective(chosen+[i]), bots[i]["id"]))
            chosen.append(choices[0])
        if len(chosen) == size:
            value = objective(chosen)
            identity = tuple(sorted(bots[i]["id"] for i in chosen))
            old_identity = tuple(sorted(bots[i]["id"] for i in best)) if best else ()
            if value > best_value or (value == best_value and identity < old_identity):
                best, best_value = chosen, value
    if best is not None:
        best.sort(key=lambda i: bots[i]["id"])
        result |= {"status": "selected_pending_manual_and_downstream_review", "objective": best_value,
                   "minimum_selected_distance": min(distance[i,j] for pos,i in enumerate(best) for j in best[pos+1:]) if size > 1 else None,
                   "selected": [bots[i] for i in best]}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=15)
    ap.add_argument("--min-maps", type=int, default=10000)
    ap.add_argument("--min-distance", type=float, default=0.05)
    ap.add_argument("--max-per-lineage", type=int, default=3)
    ap.add_argument("--min-per-count", type=int, default=500)
    a = ap.parse_args()
    catalog = json.loads(Path(a.input).read_text(encoding="utf-8"))
    result = select(catalog, a.size, a.min_maps, a.min_distance, a.max_per_lineage, a.min_per_count)
    result["catalog_sha256"] = hashlib.sha256(Path(a.input).read_bytes()).hexdigest()
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "selected": len(result["selected"]), "qualified": result["qualified"]}))
    if not result["selected"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
