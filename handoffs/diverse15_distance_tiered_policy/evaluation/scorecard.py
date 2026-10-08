"""Fixed-sample paired CTF scorecard; diagnostic calculation, not export approval.

Input: one JSON object per fixture, scalar metrics as [weighted sum, weight].
Related fixtures share cluster_id; a cluster belongs to one suite/count/stratum.
Run: python3 scorecard.py --plan scorecard_plan.json --input data.jsonl --out report.json
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def utility(scores, ego):
    if not isinstance(scores, list) or len(scores) < 2:
        raise ValueError("scores must contain at least two teams")
    if not isinstance(ego, int) or isinstance(ego, bool) or not 0 <= ego < len(scores):
        raise ValueError("invalid ego seat")
    if any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in scores):
        raise ValueError("scores must be nonnegative integers")
    own = scores[ego]
    return sum(1.0 if own > s else 0.5 if own == s else 0.0
               for i, s in enumerate(scores) if i != ego) / (len(scores) - 1)


def scalar(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name}: expected finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name}: below {minimum}")
    return float(value)


def validate_plan(plan):
    if plan.get("schema") != "ctf-scorecard-plan-1":
        raise ValueError("unsupported plan schema")
    counts = plan["team_counts"]
    if not counts or len(set(counts)) != len(counts) or any(type(n) is not int or n not in range(2, 16) for n in counts):
        raise ValueError("team_counts must be unique counts from 2 to 15")
    if type(plan["bootstrap_reps"]) is not int or plan["bootstrap_reps"] < 100:
        raise ValueError("bootstrap_reps must be an integer >=100")
    if not 0 < scalar(plan["confidence"], "confidence") < 1:
        raise ValueError("confidence must be inside (0,1)")
    for suite, weights in plan["stratum_weights_by_suite"].items():
        if not weights or any(scalar(v, f"{suite} weight") <= 0 for v in weights.values()):
            raise ValueError("stratum weights must be positive")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-10):
            raise ValueError("stratum weights must sum to one")
    for name, pair in plan.get("comparisons", {}).items():
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"invalid comparison: {name}")
    for key in ("minimum_independent_groups_total", "minimum_independent_groups_per_count"):
        if type(plan[key]) is not int or plan[key] < 1:
            raise ValueError(f"{key} must be a positive integer")


def prepare(records, plan):
    seen, cluster_locations, out = set(), {}, []
    reserved = {"play_baseline", "play_candidate", "play_gain"}
    for source in records:
        r = dict(source)
        for key in ("fixture_id", "cluster_id", "suite", "stratum"):
            if not isinstance(r.get(key), str) or not r[key]:
                raise ValueError(f"missing nonempty {key}")
        if r["fixture_id"] in seen:
            raise ValueError(f"duplicate fixture_id: {r['fixture_id']}")
        seen.add(r["fixture_id"])
        if type(r.get("teams")) is not int or r["teams"] not in plan["team_counts"]:
            raise ValueError("record count outside frozen scope")
        weights = plan["stratum_weights_by_suite"].get(r["suite"], {})
        if r["stratum"] not in weights:
            raise ValueError("record stratum outside frozen schedule")
        location = r["suite"], r["teams"], r["stratum"]
        prior = cluster_locations.setdefault(r["cluster_id"], location)
        if prior != location:
            raise ValueError("one cluster spans strata/counts/suites; redesign grouping")
        metrics = {}
        for key, pair in r.get("metrics", {}).items():
            if key in reserved or key in plan.get("comparisons", {}):
                raise ValueError(f"reserved metric: {key}")
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"{key}: expected [sum, weight]")
            num, den = scalar(pair[0], key, 0), scalar(pair[1], key, 0)
            if den == 0 and num != 0:
                raise ValueError("nonzero sum with zero weight")
            metrics[key] = (num, den)
        for gain, (base, candidate) in plan.get("comparisons", {}).items():
            if (base in metrics) != (candidate in metrics):
                raise ValueError(f"unpaired metric: {gain}")
            if base in metrics:
                if not math.isclose(metrics[base][1], metrics[candidate][1], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError("paired losses must use identical rows/weights")
                metrics[gain] = (metrics[base][0] - metrics[candidate][0], metrics[base][1])
        if ("baseline_scores" in r) != ("candidate_scores" in r):
            raise ValueError("both final score arrays are required")
        if "baseline_scores" in r:
            if len(r["baseline_scores"]) != r["teams"] or len(r["candidate_scores"]) != r["teams"]:
                raise ValueError("score array length disagrees with teams")
            b, c = utility(r["baseline_scores"], r["ego"]), utility(r["candidate_scores"], r["ego"])
            metrics |= {"play_baseline": (b, 1.0), "play_candidate": (c, 1.0), "play_gain": (c-b, 1.0)}
        if "timing" in r:
            for side in ("baseline", "candidate"):
                t = r["timing"][side]
                scalar(t["first_ms"], "first_ms", 0)
                if not isinstance(t["turn_ms"], list):
                    raise ValueError("turn_ms must be raw post-startup durations")
                for x in t["turn_ms"]:
                    scalar(x, "turn_ms", 0)
                if type(t["faults"]) is not int or t["faults"] < 0:
                    raise ValueError("faults must be a nonnegative count")
        r["metrics"] = metrics
        out.append(r)
    if not out:
        raise ValueError("no input fixtures")
    return out


def reduce_clusters(rows, metric):
    if any(metric not in r["metrics"] for r in rows):
        return None  # Never silently use only the easy/available fixtures.
    reduced = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        num, den = r["metrics"][metric]
        v = reduced[r["cluster_id"]]
        v[0] += num
        v[1] += den
    data = np.array(list(reduced.values()), dtype=np.float64)
    # Gameplay targets maps equally, after averaging that map's scheduled seats.
    if metric.startswith("play_"):
        data[:, 0] /= data[:, 1]
        data[:, 1] = 1.0
    return data


def bootstrap_ratio(data, reps, rng):
    n = len(data)
    draws = np.empty(reps)
    for lo in range(0, reps, 64):
        ids = rng.integers(n, size=(min(64, reps-lo), n))
        total = data[ids].sum(1)
        draws[lo:lo+len(ids)] = np.divide(total[:, 0], total[:, 1],
                                         out=np.full(len(ids), np.nan), where=total[:, 1] > 0)
    return draws


def summary(mean, draws, groups, confidence, complete=True):
    invalid = int((~np.isfinite(draws)).sum())
    ci = None
    # Missing/zero-label resamples must not be discarded to manufacture a CI.
    if complete and groups >= 2 and not invalid:
        tail = (1-confidence)/2
        ci = np.quantile(draws, [tail, 1-tail]).tolist()
    return {"mean": mean, "ci": ci, "groups": groups, "complete": complete,
            "invalid_bootstrap_draws": invalid,
            "degenerate_bootstrap": bool(ci is not None and ci[0] == ci[1])}


def runtime(rows):
    present = [r for r in rows if "timing" in r]
    result = {"complete": len(present) == len(rows), "fixtures_with_timing": len(present), "fixtures": len(rows)}
    if not present:
        return result
    for side in ("baseline", "candidate"):
        first = np.array([r["timing"][side]["first_ms"] for r in present])
        arrays = [np.array(r["timing"][side]["turn_ms"], dtype=float) for r in present]
        turns = np.concatenate(arrays)
        late_groups = set()
        fault_groups = set()
        for r in present:
            t = r["timing"][side]
            if t["first_ms"] > 2000 or any(ms > 50 for ms in t["turn_ms"]):
                late_groups.add(r["cluster_id"])
            if t["faults"] > 0:
                fault_groups.add(r["cluster_id"])
        groups = len({r["cluster_id"] for r in present})
        result[side] = {
            "first_max_ms": float(first.max()), "first_late": int((first > 2000).sum()),
            "post_startup_turns": len(turns), "post_startup_late": int((turns > 50).sum()),
            "post_startup_late_rate": float((turns > 50).mean()) if len(turns) else None,
            "post_startup_quantiles_ms": dict(zip(("p50", "p95", "p99", "p99_9", "max"),
                                                  np.quantile(turns, [0.5, 0.95, 0.99, 0.999, 1]).tolist())) if len(turns) else None,
            "late_cluster_count": len(late_groups), "fault_cluster_count": len(fault_groups),
            "faults": sum(r["timing"][side]["faults"] for r in present),
            "late_or_fault_cluster_rate": len(late_groups | fault_groups)/groups,
            "zero_late_cluster_upper95": 1 - 0.05 ** (1/groups) if not late_groups else None,
        }
    return result


def evaluate(records, plan):
    validate_plan(plan)
    rows = prepare(records, plan)
    reps, conf = plan["bootstrap_reps"], plan["confidence"]
    rng = np.random.default_rng(plan["bootstrap_seed"])
    report = {"schema": "ctf-scorecard-report-1", "plan": plan, "suites": {},
              "method": "fixed-sample paired cluster percentile bootstrap, stratified by frozen count/roster",
              "deployment_authorized": False,
              "limits": "Approximate intervals conditional on frozen weights and roster; no training-seed uncertainty, BT fit, hash verification or export authorization."}
    main_groups, main_min = set(), []
    for suite in sorted({r["suite"] for r in rows}):
        weights = plan["stratum_weights_by_suite"][suite]
        suite_rows = [r for r in rows if r["suite"] == suite]
        metrics = sorted({m for r in suite_rows for m in r["metrics"]}
                         | set(plan.get("comparisons", {}))
                         | {"play_baseline", "play_candidate", "play_gain"})
        sr = {"by_count": {}, "overall_equal_count": {}}
        count_boots = defaultdict(list)
        count_means = defaultdict(list)
        count_groups = defaultdict(int)
        for count in plan["team_counts"]:
            cr = [r for r in suite_rows if r["teams"] == count]
            if not cr:
                sr["by_count"][str(count)] = {"missing": True}
                if suite == "main":
                    main_min.append(0)
                continue
            play_groups = {r["cluster_id"] for r in cr if "play_gain" in r["metrics"]}
            if suite == "main":
                main_groups |= play_groups
                main_min.append(len(play_groups))
            entry = {"fixtures": len(cr), "groups": len({r['cluster_id'] for r in cr}),
                     "independent_play_groups": len(play_groups), "metrics": {}, "runtime": runtime(cr)}
            for metric in metrics:
                mean, boot, groups, complete = 0.0, np.zeros(reps), 0, True
                for stratum, weight in weights.items():
                    subset = [r for r in cr if r["stratum"] == stratum]
                    data = reduce_clusters(subset, metric) if subset else None
                    if data is None or data[:, 1].sum() <= 0:
                        complete = False
                        break
                    groups += len(data)
                    mean += weight * data[:, 0].sum() / data[:, 1].sum()
                    boot += weight * bootstrap_ratio(data, reps, rng)
                    if len(data) < 2:
                        complete = False
                point = float(mean) if complete else None
                entry["metrics"][metric] = summary(point, boot, groups, conf, complete)
                if complete:
                    count_means[metric].append(mean)
                    count_boots[metric].append(boot)
                    count_groups[metric] += groups
            sr["by_count"][str(count)] = entry
        for metric in metrics:
            complete = len(count_means[metric]) == len(plan["team_counts"])
            mean = float(np.mean(count_means[metric])) if complete else None
            draws = np.mean(count_boots[metric], axis=0) if complete else np.full(reps, np.nan)
            sr["overall_equal_count"][metric] = summary(mean, draws, count_groups[metric], conf, complete)
        report["suites"][suite] = sr
    report["independent_main_play_groups"] = len(main_groups)
    report["confirmation_sample_complete"] = bool(
        main_min and len(main_min) == len(plan["team_counts"])
        and min(main_min) >= plan["minimum_independent_groups_per_count"]
        and len(main_groups) >= plan["minimum_independent_groups_total"]
        and report["suites"].get("main", {}).get("overall_equal_count", {}).get("play_gain", {}).get("complete", False))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
    with Path(a.input).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    result = evaluate(rows, plan)
    result["provenance"] = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for name, path in (("plan_sha256", a.plan), ("input_sha256", a.input),
                                               ("calculator_sha256", __file__))}
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("independent_main_play_groups", "confirmation_sample_complete", "deployment_authorized")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
