"""Analysis of collected branches (task T0085). Every number is computed from rows; nothing is a default.

Per root with a complete row (every candidate x repeat branch complete; a root with any refused branch is excluded
and counted by reason, never imputed), U[c][r] is candidate c's terminal utility in repeat r, c = 0 the baseline:

  gain[c]               mean_r (U[c][r] - U[0][r]): the signed paired gain over the actual direct order; gain[0] = 0
  in_sample_best_gain   max_c gain[c] (>= 0). A noisy in-sample maximum: biased upward, never a demonstrated gain
  shortlist_regret[c]   max_c' mean U[c'] - mean U[c] (>= 0), relative to the best OBSERVED shortlist mean, not to a
                        true best action
  ranks                 average ranks of mean U over candidates (ties share the mean rank)
  cross_fit_gain        repeats split into two disjoint folds (even / odd repeat ids, at least two each): pick the
                        candidate with the highest fold-A gain among ALL candidates, the baseline included (a tie keeps
                        the earlier candidate, so the baseline wins ties), evaluate its gain on fold B; and the
                        reverse; the mean of the two. Unavailable with fewer than four repeats. It estimates what
                        selecting on these repeats realizes on fresh ones, not an oracle maximum
  v_spearman            rank correlation of the critic's mean leaf value per candidate with mean U (unavailable
                        when either is constant)
  continuation_faults   faults after the root, summed over the root's branches (complete branches keep them; they are
                        reported, not hidden)
  v_gate                the frozen search gate on the critic's means over the first 8 roots, the 8x4 package's tree
                        (inference/agent_v2/search.hpp gated_choice, tau): its pick, whether it overrides the
                        baseline, and that pick's gain. With the critic, gate, roots and scenarios fixed before the
                        branches, the pick does not select on the evaluation repeats, so its gain carries no repeat
                        selection bias. On an extended row, v_gate24 is the same rule over all 24 roots: a
                        counterfactual wider gate, not the frozen one
  gain_repeat_sd        per candidate, the standard deviation over repeats of the paired difference U[c][r] - U[0][r],
                        averaged over the non-baseline candidates: how much a single repeat moves a paired gain

Aggregation is per split and stratum, per exact team count, over source maps. Within a count, roots of one map are
averaged first (the cluster); maps then count equally, and counts count equally. The estimand is therefore the
equal-weight mean over counts of the equal-weight mean over maps of the stratum's sampled roots: for the
unconditioned stratum, of turns drawn uniformly from each map's eligible turns. It is not a pooled-turn mean, and the
recorded selection weights (eligible / selected turns) are reported per root, not applied. Tactical strata are
reported separately and represent only themselves. The standard error comes from the spread of map means within
each count; a count with fewer than two maps has none, and then neither has the overall estimate. Registered counts
without analysed roots are listed as missing, never averaged as zero. Intervals are normal-approximation 95%.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path

from tools.search_diagnostics.collect import load_manifest, load_roots, validate_row
from tools.search_diagnostics.schemas import SCHEMA_ANALYSIS, ValidationError, read_gz

METRICS = ("in_sample_best_gain", "cross_fit_gain", "v_gate_gain", "v_override", "v_spearman", "baseline_repeat_sd",
           "gain_repeat_sd")
GATE_ROOTS = 8


def average_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(set(a)) < 2 or len(set(b)) < 2:
        return None
    ra, rb = average_ranks(a), average_ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))


def gated_choice(mean: list[float], tau: float) -> int:
    """search.hpp gated_choice: the first highest mean if it beats mean[0] by more than tau, else 0."""
    if any(not math.isfinite(m) for m in mean):
        return 0
    best = 0
    for i in range(1, len(mean)):
        if mean[i] > mean[best]:
            best = i
    return best if best != 0 and mean[best] - mean[0] > tau else 0


def matrix(row: dict) -> tuple[list[int], list[int], dict]:
    cands = row["binding"]["candidates"]
    reps = list(range(row["binding"]["repeats"]))
    u = {(b["candidate"], b["repeat"]): b["utility"] for b in row["branches"]}
    return cands, reps, u


def root_metrics(root: dict, row: dict, tau: float) -> dict:
    cands, reps, u = matrix(row)
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    gain = {c: mean([u[c, r] - u[0, r] for r in reps]) for c in cands}
    mu = [mean([u[c, r] for r in reps]) for c in cands]
    best = max(mu)

    def pick(fold):
        g = {c: mean([u[c, r] - u[0, r] for r in fold]) for c in cands}
        return max(cands, key=lambda c: (g[c], -c))

    cross = None
    if len(reps) >= 4:
        a, b = reps[0::2], reps[1::2]
        ca, cb = pick(a), pick(b)
        cross = (mean([u[ca, r] - u[0, r] for r in b]) + mean([u[cb, r] - u[0, r] for r in a])) / 2
    vmean = [mean(root["candidates"][c]["v_values"]) for c in cands]
    gate = [c for c in cands if c < GATE_ROOTS]
    v_pick = gate[gated_choice(vmean[:len(gate)], tau)]
    v_pick24 = cands[gated_choice(vmean, tau)] if len(cands) > GATE_ROOTS else None

    def sd_of(xs):
        return math.sqrt(sum((x - mean(xs)) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else None
    paired = [sd_of([u[c, r] - u[0, r] for r in reps]) for c in cands if c != 0]
    base = [u[0, r] for r in reps]
    sd = math.sqrt(sum((x - mean(base)) ** 2 for x in base) / (len(base) - 1)) if len(base) > 1 else None
    return {
        "root_id": root["root_id"], "teams": root["source"]["teams"], "map": root["source"]["match_id"],
        "stratum": root["selection"]["stratum"], "candidates": len(cands), "repeats": len(reps),
        "gain": [gain[c] for c in cands], "mean_utility": mu, "ranks": average_ranks(mu),
        "shortlist_regret": [best - m for m in mu], "in_sample_best_gain": max(gain.values()),
        "cross_fit_gain": cross, "v_mean": vmean, "v_spearman": spearman(vmean, mu),
        "v_gate_pick": v_pick, "v_override": float(v_pick != 0), "v_gate_gain": gain[v_pick],
        "baseline_repeat_sd": sd, "context": root.get("context"),
        "gain_repeat_sd": None if not paired or None in paired else sum(paired) / len(paired),
        "v_gate24_pick": v_pick24, "v_gate24_gain": gain[v_pick24] if v_pick24 is not None else None,
        "selection_weight": root["selection"]["weight"],
        "continuation_faults": sum(len(b["faults"]["continuation"]) for b in row["branches"]),
    }


def aggregate(rows: list[dict], metric: str, counts: list[int] | None = None) -> dict:
    by_count: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    for m in rows:
        if m[metric] is None:
            missing += 1
            continue
        by_count[m["teams"]][m["map"]].append(m[metric])
    per_count = {}
    for n, maps in sorted(by_count.items()):
        xs = [sum(v) / len(v) for v in maps.values()]
        mean = sum(xs) / len(xs)
        se = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else None
        per_count[str(n)] = {"maps": len(xs), "roots": sum(len(v) for v in maps.values()), "mean": mean, "se": se}
    absent = sorted(set(counts or []) - {int(n) for n in per_count})
    if not per_count:
        return {"status": "unavailable: no roots with this metric", "missing_roots": missing, "missing_counts": absent}
    vals = list(per_count.values())
    mean = sum(v["mean"] for v in vals) / len(vals)
    se = None if any(v["se"] is None for v in vals) else math.sqrt(sum(v["se"] ** 2 for v in vals)) / len(vals)
    return {"status": "computed", "counts": len(vals), "mean_equal_count_weight": mean, "se": se,
            "ci95_normal": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            "se_status": "computed" if se is not None else "unavailable: a count has fewer than two maps",
            "missing_roots": missing, "missing_counts": absent,
            "scope": "the analysed counts only" if absent else "every registered count", "per_count": per_count}


def analyze(roots_dir: Path, rows_dir: Path, tau: float = 0.002, splits: dict | None = None,
            counts: list[int] | None = None) -> dict:
    man = load_manifest(rows_dir)
    roots = load_roots(roots_dir)
    excluded: Counter = Counter()
    per_root, seen = [], set()
    for p in sorted(Path(rows_dir).glob("n[0-9][0-9]/*.json.gz")):
        rid = p.name[:-len(".json.gz")]
        if rid in seen:
            raise ValidationError(f"duplicate row for root {rid}")
        seen.add(rid)
        root = roots.get(rid)
        if root is None:
            raise ValidationError(f"row {rid} has no root record")
        row = validate_row(read_gz(p), man["identity_digest"], root, man["candidate_set"], man["repeats"],
                           man["scenario_seed"])
        refused = [b["reason"] for b in row["branches"] if b["status"] != "complete"]
        if refused:
            excluded[refused[0].split(":")[0]] += 1
            continue
        m = root_metrics(root, row, tau)
        m["split"] = (splits or {}).get(root["source"]["match_id"], "unsplit")
        per_root.append(m)
    excluded["not collected"] += len(set(roots) - seen)
    groups = defaultdict(list)
    for m in per_root:
        groups[(m["split"], m["stratum"])].append(m)
    return {
        "schema": SCHEMA_ANALYSIS, "tau": tau, "collection": man["identity_digest"], "evidence": man["evidence"],
        "candidate_set": man["candidate_set"], "roots": len(roots), "analysed": len(per_root),
        "excluded": {k: v for k, v in excluded.items() if v},
        "groups": {f"{s}/{st}": {metric: aggregate(ms, metric, counts) for metric in METRICS}
                   for (s, st), ms in sorted(groups.items())},
        "per_root": per_root,
        "notes": __doc__.split("\n\n", 1)[1],
    }
