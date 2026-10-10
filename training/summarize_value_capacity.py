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
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion.paths import REPO, sha256
from training.evaluate_value import clustered_mean_ci

FIELDS = re.compile(r"(\w+) (-?[0-9.]+(?:e-?\d+)?)")


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
    timings, counters, turns, coverage = defaultdict(list), defaultdict(lambda: defaultdict(float)), defaultdict(int), defaultdict(lambda: defaultdict(int))
    for path in sorted((directory / "matches").rglob("*.json.gz")):
        arm = keys.get(path.name.split("-")[-1].split(".")[0])
        if arm is None:
            continue
        match = json.loads(gzip.decompress(path.read_bytes()))
        for seat, bot in zip(match["seats"], match["bots"]):
            if not seat["name"].startswith("@candidate"):
                continue
            timings[arm].append({k: bot.get(k) for k in ("first_ms", "p50_ms", "p99_ms", "max_ms", "faults", "late", "dead")})
            turns[arm] += len(match["blocks"])
            for line in bot.get("stderr_tail", []):
                if match_line := re.match(r"^(SEARCH|VALUE|VBUDGET) (.*)$", line.strip()):
                    prefix = match_line[1].lower()
                    coverage[arm][prefix] += 1
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
              "note": "p50/p99 summaries are per-game full replies, not pooled quantiles. One map/count cannot estimate per-count uncertainty. No strength conclusion."}
    by_arm = {arm: {r["fixture_id"]: r for r in rows if r["arm"] == arm} for arm in arms}
    for arm in arms:
        rs, ts = list(by_arm[arm].values()), timings[arm]
        t = {}
        for name in ("first_ms", "p50_ms", "p99_ms", "max_ms"):
            values = [r[name] for r in ts if r[name] is not None]
            t[name] = {"median": float(np.median(values)), "max": max(values)} if values else None
        c = dict(counters[arm])
        if coverage[arm].get("value"):
            c["value.calls_per_turn"] = sum(c.get(f"value.{n}_calls", 0) for n in ("refine", "beam", "rollout")) / turns[arm]
            c["value.ms_per_turn"] = 1000 * c.get("value.seconds", 0) / turns[arm]
        faults = sum(r["faults"] for r in rs)
        complete = len(rs) == expected and len(ts) == len(rs)
        deadline = bool(complete and faults == 0 and not any(r.get("dead") or r.get("late") for r in rs)
                        and t["first_ms"] and t["max_ms"] and t["first_ms"]["max"] <= 2000 and t["max_ms"]["max"] <= 50)
        result["arms"][arm] = {"games": len(rs), "utility": float(np.mean([r["utility"] for r in rs])), "faults": faults,
                              "late": sum(r.get("late", 0) for r in rs), "timing": t, "counters": c,
                              "counter_games": dict(coverage[arm]), "deadline_passes_observed_games": deadline}
    for arm in arms:
        baseline = arm.removesuffix("~aa") if arm.endswith("~aa") else f"cap_off_{work}"
        if arm == baseline or baseline not in by_arm:
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
    a = ap.parse_args(argv)
    out, build = Path(a.out), Path(a.build)
    validation = json.loads((out / "validation.json").read_text())
    report = {"generated_by": "python -m training.summarize_value_capacity", "default_changed": False,
              "implemented": True, "trained": True, "training_seeds": [1], "target": "mixed-policy exploratory",
              "decision_improvement_demonstrated": False, "game_strength_improvement_demonstrated": False,
              "validation": validation, "integer": {}, "screens": {}}
    lines = ["# 5XL capacity experiment", "", "Generated by `python -m training.summarize_value_capacity`. Production is unchanged. "
             "This is a mixed-policy exploratory pilot with one training seed; no decision or game-strength improvement is demonstrated.", "",
             "| Model | Parameters | Validation log loss | Brier | Epochs / best | Optimizer steps | Fit / pipeline seconds |",
             "|---|---:|---:|---:|---|---:|---| "]
    for name, row in validation["runs"].items():
        e = row["effort"]
        lines.append(f"| {name} | {e['params']} | {row['all']['logloss']:.6f} | {row['all']['brier']:.6f} | {e['epochs']} / {e['best_epoch']} | "
                     f"{e.get('optimizer_steps')} | {e.get('fit_seconds')} / {e.get('pipeline_seconds')} |")
        run = REPO / ("runs/reference_v3xl" if name == "Reference" else f"runs/v_capacity_5xl/{name}_s1")
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
            lines.append(f"| {arm} | {r['games']} | {r['utility']:.4f} | {r['faults']} / {r['late']} | {t['first_ms']['max'] if t['first_ms'] else None} | "
                         f"{t['p50_ms']['median'] if t['p50_ms'] else None} / {t['p99_ms']['max'] if t['p99_ms'] else None} / {t['max_ms']['max'] if t['max_ms'] else None} | "
                         f"{c.get('value.calls_per_turn')} / {c.get('value.ms_per_turn')} |")
        lines += ["", "Paired utility differences (A/A against its identical base; others against V off):", ""]
        for comparison, interval in s.get("paired", {}).items():
            lines.append(f"- {comparison}: {interval['mean']:+.5f}, [{interval['lo']}, {interval['hi']}], {interval['clusters']} map groups; {interval['verdict']}.")
        source = build / f"screen_{work}_analysis" / "RUN.json"
        if source.is_file():
            report["screens"][work]["run"] = json.loads(source.read_text())
    report["cpu_deadline_verified"] = {
        work: {arm: r["deadline_passes_observed_games"] for arm, r in screen.get("arms", {}).items()}
        for work, screen in report["screens"].items()}
    lines += ["", "## Evidence status and continuation", "",
              "Implemented and trained. Trained integer verification is recorded above. CPU deadlines apply only to the observed games and load; "
              "inspect missing counter coverage, fallbacks and displaced samples in SUMMARY.json. No decision or game-strength improvement is established. "
              "The tiny screen and A/A reruns are insufficient for deployment confirmation.", "",
              "The best 5XL and student checkpoints stay under runs/v_capacity_5xl/, and exports/bundles stay under build/v_capacity_5xl/. "
              "Next: specified frozen C08 continuation data with bounded complete-order alternatives and repeated real continuations, "
              "three training seeds for any selected finalists, validation-only tuning, then the repository's fresh 10,000-map confirmation. "
              "Commands and seed restrictions are in docs/V_CAPACITY_5XL.md. Final test labels remain unscored."]
    (out / "SUMMARY.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "SUMMARY.md")


if __name__ == "__main__":
    main()
