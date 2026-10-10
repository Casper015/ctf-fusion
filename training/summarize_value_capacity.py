"""Regenerate a conservative capacity evidence summary from pipeline outputs and kept host match logs.

python -m training.summarize_value_capacity --build build/v_capacity_5xl --out reports/models/v_capacity_5xl

The existing screen analyzer is also run. With one map per count its within-count
variance is unestimable; this supplement computes paired differences over the
independent map groups with a small-sample t multiplier and never approves a bot.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from fusion.paths import REPO, sha256
from fusion.protocol import ATTACK, DROP, PICKUP, WAIT, parse_reply
from training.dataset import split_of
from training.evaluate_value import clustered_mean_ci

FIELDS = re.compile(r"(\w+) (-?[0-9.]+(?:e-?\d+)?)")


def branch_summary(directory: Path, rank: Path) -> dict:
    files = sorted(directory.rglob("*.json.gz"))
    if not files:
        return {"status": "not run"}
    roots, repetitions, scope = [], Counter(), Counter()
    mismatches, faults, prefix_faults = [], 0, 0
    for path in files:
        row = json.loads(gzip.decompress(path.read_bytes()))
        entry = {k: row[k] for k in ("match_id", "teams", "turn")}
        entry["sha256"] = sha256(path)
        if "dropped" in row:
            entry["dropped"] = row["dropped"]
            roots.append(entry)
            continue
        entry |= {k: row[k] for k in ("map_seed", "carrier_sha256", "candidate_mode", "candidates")}
        entry["split"] = split_of(f"map-{row['map_seed']}")
        by_cand = defaultdict(list)
        for branch in row["branches"]:
            by_cand[branch["candidate"]].append(branch["outcome"])
            mismatches.append(branch["prefix_mismatch"])
            faults += len(branch["faults"])
            prefix_faults += branch["prefix_faults"]
        repetitions.update(len(outcomes) for outcomes in by_cand.values())
        entry["continuation_outcomes"] = dict(by_cand)
        original = parse_reply(row["candidates"][0], row["turn"])
        for line in row["candidates"][1:]:
            order = parse_reply(line, row["turn"])
            if order is None:
                raise ValueError("malformed complete-order candidate")
            moves = sum(a.move != b.move for a, b in zip(original, order))
            scope["single_unit_movement"] += moves == 1
            scope["coordinated_movement"] += moves >= 2
            scope["attack_allocation"] += any(b.act == ATTACK and (a.act != b.act or a.target != b.target) for a, b in zip(original, order))
            scope["pickup_wait_change"] += any(b.act in (WAIT, PICKUP) and a.act != b.act for a, b in zip(original, order))
            scope["drop_plan"] += any(b.act == DROP for b in order)
        roots.append(entry)
    valid = [row for row in roots if "dropped" not in row]
    return {"status": "isolated exploratory real continuations, not model selection", "roots": roots,
            "successful_roots": len(valid), "dropped_roots": len(roots) - len(valid),
            "independent_map_groups": len({row['map_seed'] for row in valid}),
            "repetitions_per_candidate": dict(repetitions), "continuations": sum(repetitions.elements()),
            "faults": faults, "prefix_faults": prefix_faults,
            "prefix_mismatch": {"min": min(mismatches), "max": max(mismatches), "mean": float(np.mean(mismatches))} if mismatches else None,
            "order_scope_counts": dict(scope), "scope_note": "Overlapping descriptions relative to the on-policy order; available legal actions depend on each root.",
            "rank_index": json.loads((rank / "index.json").read_text()) if (rank / "index.json").is_file() else None,
            "test_usage": "No model predictions or selection on the branch test split; original final test cache labels remain unscored."}


def screen_summary(work: str, build: Path) -> dict:
    config = json.loads((build / f"screen_{work}.json").read_text())
    directory = REPO / config["work_dir"]
    path = directory / "results.jsonl"
    if not path.is_file():
        return {"status": "not run"}
    identities = json.loads((directory / "identities.json").read_text())
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(r.get("identity") != identities.get(r["arm"], {}).get("digest") for r in rows):
        raise ValueError("screen contains stale identities; refuse to summarize")
    if len({(r["arm"], r["fixture_id"]) for r in rows}) != len(rows):
        raise ValueError("duplicate arm/fixture rows; refuse to summarize")
    expected = len(json.loads((directory / "schedule.json").read_text())["fixtures"])
    arms = sorted({r["arm"] for r in rows})
    keys = {hashlib.sha256(arm.encode()).hexdigest()[:10]: arm for arm in arms}
    timings, counters = defaultdict(list), defaultdict(lambda: defaultdict(float))
    turns, coverage = defaultdict(lambda: defaultdict(int)), defaultdict(lambda: defaultdict(int))
    for path in sorted((directory / "matches").rglob("*.json.gz")):
        arm = keys.get(path.name.split("-")[-1].split(".")[0])
        if arm is None:
            continue
        match = json.loads(gzip.decompress(path.read_bytes()))
        for seat, bot in zip(match["seats"], match["bots"]):
            if not seat["name"].startswith("@candidate"):
                continue
            timings[arm].append({k: bot.get(k) for k in ("first_ms", "p50_ms", "p99_ms", "max_ms", "faults", "late", "dead")})
            for line in bot.get("stderr_tail", []):
                if match_line := re.match(r"^(SEARCH|VALUE|VBUDGET) (.*)$", line.strip()):
                    prefix = match_line[1].lower()
                    coverage[arm][prefix] += 1
                    turns[arm][prefix] += len(match["blocks"])
                    for name, value in FIELDS.findall(match_line[2]):
                        key = f"{prefix}.{name}"
                        if name.startswith("coef_") or name in ("fixed", "max_calls", "deadline_ms"):
                            counters[arm][key] = float(value)
                        else:
                            counters[arm][key] += float(value)
    result = {"status": "exploratory only", "rows_sha256": sha256(directory / "results.jsonl"),
              "config": config, "identities": identities,
              "schedule": json.loads((directory / "schedule.json").read_text()),
              "expected_games_per_arm": expected,
              "independent_map_groups": len({r["map_seed"] for r in rows}), "arms": {}, "paired": {},
              "note": "p50/p99 summaries are per-game full replies, not pooled quantiles. Counter rates use only covered games; missing counters remain missing. One map/count cannot estimate per-count uncertainty. No strength conclusion."}
    by_arm = {arm: {r["fixture_id"]: r for r in rows if r["arm"] == arm} for arm in arms}
    for arm in arms:
        rs, ts = list(by_arm[arm].values()), timings[arm]
        t = {}
        for name in ("first_ms", "p50_ms", "p99_ms", "max_ms"):
            values = [r[name] for r in ts if r[name] is not None]
            t[name] = {"median": float(np.median(values)), "max": max(values)} if values else None
        c = dict(counters[arm])
        if coverage[arm].get("value"):
            c["value.calls_per_turn"] = sum(c.get(f"value.{n}_calls", 0) for n in ("refine", "beam", "rollout")) / turns[arm]["value"]
            c["value.ms_per_turn"] = 1000 * c.get("value.seconds", 0) / turns[arm]["value"]
        if coverage[arm].get("search"):
            c["search.samples_per_turn"] = c["search.samples"] / turns[arm]["search"]
        faults = sum(r["faults"] for r in rs)
        complete = len(rs) == expected and len(ts) == len(rs)
        deadline = bool(complete and faults == 0 and not any(r.get("dead") or r.get("late") for r in rs)
                        and t["first_ms"] and t["max_ms"] and t["first_ms"]["max"] <= 2000 and t["max_ms"]["max"] <= 50)
        result["arms"][arm] = {"games": len(rs), "utility": float(np.mean([r["utility"] for r in rs])), "faults": faults,
                              "late": sum(r.get("late", 0) for r in rs), "timing": t, "counters": c,
                              "counter_games": dict(coverage[arm]), "counter_turns": dict(turns[arm]),
                              "deadline_passes_observed_games": deadline}
    comparisons = config.get("comparisons", [[arm, f"cap_off_{work}"] for arm in arms if not arm.endswith("~aa")])
    comparisons += [[arm, arm.removesuffix("~aa")] for arm in arms if arm.endswith("~aa")]
    for arm, baseline in comparisons:
        if arm == baseline or arm not in by_arm or baseline not in by_arm:
            continue
        common = sorted(set(by_arm[arm]) & set(by_arm[baseline]))
        x = np.array([by_arm[arm][k]["utility"] - by_arm[baseline][k]["utility"] for k in common])
        maps = np.array([by_arm[arm][k]["map_seed"] for k in common])
        k = len(np.unique(maps))
        if k < 2:
            interval = {"mean": float(x.mean()) if len(x) else None, "lo": None, "hi": None, "clusters": k, "verdict": "not shown"}
        else:
            critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(k, 2.262 if k < 31 else 1.96)
            interval = clustered_mean_ci(x, np.ones(len(x)), maps, z=critical)
        result["paired"][f"{arm}:{baseline}"] = interval
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", default="build/v_capacity_5xl")
    ap.add_argument("--out", default="reports/models/v_capacity_5xl")
    ap.add_argument("--branches", default="data/branches/v_capacity_5xl_complete_isolated")
    ap.add_argument("--branch-rank", default="data/rank/v_capacity_5xl_complete_isolated")
    a = ap.parse_args(argv)
    out, build = Path(a.out), Path(a.build)
    validation = json.loads((out / "validation.json").read_text())
    report = {"generated_by": "python -m training.summarize_value_capacity", "default_changed": False,
              "generator_sha256": sha256(Path(__file__)),
              "implemented": True, "trained": True, "training_seeds": [1], "target": "mixed-policy exploratory",
              "decision_improvement_demonstrated": False, "game_strength_improvement_demonstrated": False,
              "validation": validation, "training_pipeline": {}, "integer": {}, "screens": {}}
    lines = ["# 5XL capacity experiment", "", "Generated by `python -m training.summarize_value_capacity`. Production is unchanged. "
             "This is a mixed-policy exploratory pilot with one training seed; no decision or game-strength improvement is demonstrated.", "",
             "| Model | Parameters | Validation log loss | Brier | Epochs / best | Optimizer steps | Fit / pipeline seconds |",
             "|---|---:|---:|---:|---|---:|---| "]
    for name, row in validation["runs"].items():
        e = row["effort"]
        run = REPO / ("runs/reference_v3xl" if name == "Reference" else f"runs/v_capacity_5xl/{name}_s1")
        distill = run / "distill_pipeline.json"
        complete_pipeline = json.loads(distill.read_text())["pipeline_seconds"] if distill.is_file() else e.get("pipeline_seconds")
        report["training_pipeline"][name] = {"fit_seconds": e.get("fit_seconds"), "model_pipeline_seconds": e.get("pipeline_seconds"),
                                            "complete_pipeline_seconds": complete_pipeline,
                                            "distillation": json.loads(distill.read_text()) if distill.is_file() else None}
        lines.append(f"| {name} | {e['params']} | {row['all']['logloss']:.6f} | {row['all']['brier']:.6f} | {e['epochs']} / {e['best_epoch']} | "
                     f"{e.get('optimizer_steps')} | {e.get('fit_seconds')} / {complete_pipeline} |")
        destination = out / "runs" / name
        destination.mkdir(parents=True, exist_ok=True)
        for file in ("config.json", "log.jsonl", "best.json", "distill_pipeline.json"):
            if (run / file).is_file():
                shutil.copyfile(run / file, destination / file)
        path = out / f"integer_{name}.json"
        if path.is_file():
            report["integer"][name] = json.loads(path.read_text())
    report["integer_export_verified"] = all(name in report["integer"] for name in ("XL", "4XL", "5XL", "Student"))
    comparison = next(c for c in validation["compare"] if c["x"] == "5XL" and c["y"] == "XL")
    loss, regret = comparison["metrics"]["logloss"], comparison["metrics"].get("regret")
    prediction = (f"5XL − XL paired validation log loss: {loss['mean']:+.6f}, "
                  f"95% interval [{loss['lo']:+.6f}, {loss['hi']:+.6f}], {loss['clusters']} map groups; {loss['verdict']}. ")
    if regret:
        prediction += (f"Paired selection regret: {regret['mean']:+.6f} [{regret['lo']:+.6f}, {regret['hi']:+.6f}], "
                       f"{regret['clusters']} map groups; {regret['verdict']}. ")
    lines += ["", prediction + "See validation.json/markdown for all metrics, pair baselines, per-count/phase scores and incumbent comparisons.", "",
              "## Candidate-pair accuracy", "", "Resolved pairs only; ties in model predictions count as half correct. All labels below have one continuation per candidate.", "",
              "| Model | Accuracy / 95% interval | Independent maps | Random / majority baseline | Versus random |", "|---|---|---:|---|---|"]
    for name, row in validation["runs"].items():
        rank = row.get("rank")
        if rank:
            acc = rank["pair_accuracy"]
            lines.append(f"| {name} | {acc['mean']:.4f} / [{acc['lo']:.4f}, {acc['hi']:.4f}] | {acc['clusters']} | "
                         f"{rank['pair_random_baseline']:.4f} / {rank['pair_majority_baseline']:.4f} | {acc['verdict']} |")
    lines += ["",
              "## Trained integer exports", "", "| Model | Native/Python parity rows | Max / mean probability error | Header plus chunks bytes | Largest chunk bytes |",
              "|---|---|---|---:|---:|"]
    for name, r in report["integer"].items():
        error = r["quantization_error"]
        lines.append(f"| {name} | {r['parity_rows']} / {r['python_parity_rows']} | {error['max_probability']:.6f} / {error['mean_probability']:.6f} | "
                     f"{r['header_bytes'] + r['chunk_bytes']} | {r['largest_chunk_bytes']} |")
    cost = out / "value_cost_DESKTOP-G60OJUD.json"
    if cost.is_file():
        report["value_cost"] = json.loads(cost.read_text())
        lines += ["", "## Isolated and concurrent V cost", "", "One V call per turn, actual encoded streams, worst copy's p99. These are V-only costs; full replies follow below.", "",
                  "| Model | Concurrent copies | Full p99 ms | Incremental p99 ms | First-layer share |", "|---|---:|---:|---:|---:|"]
        for name, row in report["value_cost"]["sets"].items():
            for copies, metrics in row["copies"].items():
                lines.append(f"| {Path(name).name} | {copies} | {metrics['turn_full_ms']['p99']:.3f} | {metrics['turn_incremental_ms']['p99']:.3f} | {metrics['first_layer_share']:.3f} |")
    for work in ("time", "work"):
        s = screen_summary(work, build)
        report["screens"][work] = s
        lines += ["", f"## Host {work} screen", "", s.get("note", s["status"]), "",
                  "| Arm | Games | Utility | Faults / late | First max ms | Reply p50 median / p99 worst / max ms | V calls / ms per turn |",
                  "|---|---:|---:|---|---:|---|---|"]
        for arm, r in s.get("arms", {}).items():
            t, c = r["timing"], r["counters"]
            def fmt(value):
                return f"{value:.3f}" if value is not None else "missing"
            lines.append(f"| {arm} | {r['games']} | {r['utility']:.4f} | {r['faults']} / {r['late']} | {fmt(t['first_ms']['max'] if t['first_ms'] else None)} | "
                         f"{fmt(t['p50_ms']['median'] if t['p50_ms'] else None)} / {fmt(t['p99_ms']['max'] if t['p99_ms'] else None)} / {fmt(t['max_ms']['max'] if t['max_ms'] else None)} | "
                         f"{fmt(c.get('value.calls_per_turn'))} / {fmt(c.get('value.ms_per_turn'))} |")
        lines += ["", "Covered games only (counters are absent for the two-team match):", "",
                  "| Arm | SEARCH / VALUE / VBUDGET games | Search samples / covered turn | Truncated searches | Refine cuts | Skipped V calls / fallback decisions |",
                  "|---|---|---:|---:|---:|---|"]
        for arm, r in s.get("arms", {}).items():
            c, coverage = r["counters"], r["counter_games"]
            lines.append(f"| {arm} | {' / '.join(str(coverage.get(k, 0)) for k in ('search', 'value', 'vbudget'))} | "
                         f"{fmt(c.get('search.samples_per_turn'))} | {c.get('search.truncated', 'missing')} | {c.get('search.refine_cuts', 'missing')} | "
                         f"{c.get('vbudget.skipped_calls', 'missing')} / {c.get('vbudget.fallback_decisions', 'missing')} |")
        lines += ["", "Configured paired utility differences and A/A against its identical base:", ""]
        for comparison, interval in s.get("paired", {}).items():
            lines.append(f"- {comparison}: {interval['mean']:+.5f}, [{interval['lo']}, {interval['hi']}], {interval['clusters']} map groups; {interval['verdict']}.")
        source = build / f"screen_{work}_analysis" / "RUN.json"
        if source.is_file():
            report["screens"][work]["run"] = json.loads(source.read_text())
    report["cpu_deadline_verified"] = {
        work: {arm: r["deadline_passes_observed_games"] for arm, r in screen.get("arms", {}).items()}
        for work, screen in report["screens"].items()}
    branch = branch_summary(Path(a.branches), Path(a.branch_rank))
    report["complete_order_branches"] = branch
    (out / "complete_order_branches.json").write_text(json.dumps(branch, indent=1) + "\n", encoding="utf-8")
    lines += ["", "## Bounded complete-order continuations", "", branch["status"] + "."]
    if branch.get("successful_roots"):
        lines += ["", f"{branch['successful_roots']} successful roots in {branch['independent_map_groups']} independent maps; "
                  f"{branch['dropped_roots']} dropped roots. {branch['continuations']} real continuations, "
                  f"repetitions per candidate: {branch['repetitions_per_candidate']}. "
                  f"Faults {branch['faults']}, prefix faults {branch['prefix_faults']}, prefix mismatch {branch['prefix_mismatch']}. "
                  "The frozen continuation carrier is the matched C08/p2-XXL/XL binary with the recorded eight-call / 20 ms bound. "
                  "Two repeats are noisy; raw outcomes and order coverage are in complete_order_branches.json. "
                  "These roots do not select a model. Branch map splits are retained; test predictions remain unscored."]
        if branch["rank_index"]:
            lines += ["", f"Rank label counts by split: `{branch['rank_index']['stats']}`."]
            if not branch["rank_index"]["stats"].get("val", {}).get("resolved_pairs", 0):
                lines += ["There are no resolved validation pairs, so this collection cannot establish improved candidate ranking."]
    lines += ["", "## Evidence status and continuation", "",
              "Implemented and trained. Trained integer verification is recorded above. CPU deadlines apply only to the observed games and load; "
              "inspect missing counter coverage, fallbacks and displaced samples in SUMMARY.json. No decision or game-strength improvement is established. "
              "The tiny screen and A/A reruns are insufficient for deployment confirmation.", "",
              "The best 5XL and student checkpoints stay under runs/v_capacity_5xl/, and exports/bundles stay under build/v_capacity_5xl/. "
              "Next: expand specified frozen C08 continuation training data with bounded complete-order alternatives and repeated real continuations, "
              "three training seeds for any selected finalists, validation-only tuning, then the repository's fresh 10,000-map confirmation. "
              "Commands and seed restrictions are in docs/V_CAPACITY_5XL.md. Final test labels remain unscored."]
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "SUMMARY.md")


if __name__ == "__main__":
    main()
