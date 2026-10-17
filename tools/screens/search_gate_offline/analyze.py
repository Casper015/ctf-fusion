"""Apply only section 1 of docs/AGENT_V2_SEARCH_GATE.md to fixed-tree receipts (T0076)."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "bots/agent_v2/search_gate/diagnostics"
GRID = (0, .001, .002, .005, .01, .015, .02, .03, .05, .075, .1, .15, .2)
CONFIGS = {"N": (4, 2), "W": (8, 4)}
GAMES = {"n02": 2, "n04": 4, "n15": 15}
BOOTSTRAPS = 2000
BOOTSTRAP_SEED = int.from_bytes(hashlib.sha256(b"20262120|bootstrap").digest()[:8], "little")


def splitmix64(x: int) -> int:
    mask = (1 << 64) - 1
    x = (x + 0x9E3779B97F4A7C15) & mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


SALTS = [str(splitmix64(20262120 + r)) for r in range(4)]
REF = str(splitmix64(20262120 + 1000))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_text(value, depth=0) -> str:
    """Sorted keys and a fixed, round-trippable 17-significant-digit float formatter."""
    if isinstance(value, dict):
        if not value:
            return "{}"
        body = ["  " * (depth + 1) + json.dumps(k) + ": " + json_text(v, depth + 1)
                for k, v in sorted(value.items())]
        return "{\n" + ",\n".join(body) + "\n" + "  " * depth + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(json_text(v, depth) for v in value) + "]"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite report number")
        return format(value, ".17g")
    return json.dumps(value, allow_nan=False)


def read_states(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("empty tree file")
    seen = set()
    for row in rows:
        game, seat, turn = row["game"], row["seat"], row["turn"]
        key = game, seat, turn
        if key in seen:
            raise ValueError(f"duplicate state {key}")
        seen.add(key)
        if (game not in GAMES or row["teams"] != GAMES[game] or row["me"] != seat or
                not 0 <= seat < GAMES[game] or not 1 <= turn <= 400 or
                row["split"] != ("validation" if game == "n04" else "selection")):
            raise ValueError(f"invalid state identity {key}")
        roots = row["roots"]
        if not 1 <= len(roots) <= 8 or roots[0] != row["actor"] or len(set(roots)) != len(roots):
            raise ValueError(f"invalid roots {key}")
        trees = {t["salt"]: t for t in row["trees"]}
        if len(row["trees"]) != 5 or set(trees) != set([*SALTS, REF]):
            raise ValueError(f"repeat/reference salt mismatch {key}")
        for salt, tree in trees.items():
            samples = 16 if salt == REF else 4
            values = np.asarray(tree["v"], dtype=np.float64)
            terminal = np.asarray(tree["terminal"])
            error = tree["max_abs_f64"]
            if (tree["S"] != samples or values.shape != (len(roots), samples) or terminal.shape != values.shape or
                    not np.isfinite(values).all() or not np.isin(terminal, [0, 1]).all() or
                    not math.isfinite(error) or error < 0):
                raise ValueError(f"invalid tree {key}, salt {salt}")
        row["by_salt"] = trees
    return sorted(rows, key=lambda row: (row["game"], row["seat"], row["turn"]))


def sequential(values) -> float:
    """Left-to-right double additions, as Search::choose sums (Python 3.12+ sum() of floats is compensated)."""
    total = 0.0
    for v in values:
        total += v
    return total


def arrays(rows: list[dict], candidates: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(rows), 4, candidates), np.nan, dtype=np.float64)
    reference = np.zeros((len(rows), candidates), dtype=np.float64)
    for si, row in enumerate(rows):
        nr = min(candidates, len(row["roots"]))
        for r, salt in enumerate(SALTS):
            values = row["by_salt"][salt]["v"]
            for i in range(nr):
                # Search::choose accumulates in scenario order, in double.
                means[si, r, i] = sequential(float(v) for v in values[i][:samples]) / samples
        vref = row["by_salt"][REF]["v"]
        for i in range(nr):
            reference[si, i] = sequential(float(vref[i][s]) - float(vref[0][s]) for s in range(16)) / 16
    return means, reference


def best_roots(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Strict comparisons preserve the first highest root, including actor ties.
    best = np.zeros(means.shape[:2], dtype=np.int64)
    top = means[:, :, 0].copy()
    for i in range(1, means.shape[2]):
        higher = means[:, :, i] > top
        best[higher] = i
        top = np.where(higher, means[:, :, i], top)
    return best, top - means[:, :, 0]


def gated_picks(means: np.ndarray, tau: float) -> np.ndarray:
    best, margin = best_roots(means)
    return np.where((best != 0) & (margin > tau), best, 0)


def ci95(values: np.ndarray) -> list[float] | None:
    # C is undefined in a resample with no overrides; retain that count in the receipts.
    valid = values[np.isfinite(values)]
    return [float(x) for x in np.percentile(valid, [2.5, 97.5])] if valid.size else None


def metrics(means: np.ndarray, reference: np.ndarray, tau: float, eps: float,
            resamples: np.ndarray | None) -> dict:
    picks = gated_picks(means, tau)
    n = len(means)
    gain = np.take_along_axis(reference, picks, axis=1) if n else np.empty((0, 4))
    overrides = picks != 0
    n_over_state = overrides.sum(axis=1)
    confirmed_state = (overrides & (gain > 0)).sum(axis=1)
    gain_state = gain.sum(axis=1)
    n_over = int(n_over_state.sum())
    result = {"tau": tau, "states": n, "pairs": 4 * n, "n_over": n_over,
              "override_rate": n_over / (4 * n) if n else None,
              "C": float(confirmed_state.sum() / n_over) if n_over else None,
              "G": float(gain_state.sum() / (4 * n)) if n else None,
              "C_ci95": None, "G_ci95": None, "C_undefined_bootstraps": None, "eligible": False}
    if resamples is not None and n:
        over_boot = n_over_state[resamples].sum(axis=1)
        confirm_boot = confirmed_state[resamples].sum(axis=1)
        c_boot = np.divide(confirm_boot, over_boot, out=np.full(BOOTSTRAPS, np.nan), where=over_boot != 0)
        g_boot = gain_state[resamples].sum(axis=1) / (4 * n)
        result.update(C_ci95=ci95(c_boot), G_ci95=ci95(g_boot), C_undefined_bootstraps=int((over_boot == 0).sum()))
    lower = result["C_ci95"][0] if result["C_ci95"] is not None else None
    result["eligible"] = bool(tau >= 2 * eps and n_over >= 30 and lower is not None and lower >= .75)
    result["eligible_checks"] = {"tau_ge_2eps": tau >= 2 * eps, "n_over_ge_30": n_over >= 30,
                                 "C_lower_ge_0.75": lower is not None and lower >= .75}
    return result


def choose(selection: dict) -> tuple[dict | None, dict]:
    eligible = {name: next((row for row in selection[name] if row["eligible"]), None) for name in CONFIGS}
    supported = [name for name in CONFIGS if eligible[name] is not None]
    decision = {"tau_c": {name: eligible[name]["tau"] if eligible[name] is not None else None for name in CONFIGS},
                "supported": supported}
    if not supported:
        return None, dict(decision, step="selection step 5", reason="neither configuration has an eligible tau")
    if supported == ["W"]:
        breadth, reason = "W", "N unsupported; W supported"
    elif "W" not in supported:
        breadth, reason = "N", "W unsupported; N supported"
    else:
        delta = eligible["W"]["G"] - eligible["N"]["G"]
        decision["G_W_minus_N"] = delta
        breadth = "W" if delta >= .0001 else "N"
        reason = "larger selection G" if abs(delta) >= .0001 else "G difference under 0.0001; choose N"
    k, s = CONFIGS[breadth]
    choice = {"breadth": breadth, "candidates": k, "samples": s, "tau": eligible[breadth]["tau"]}
    return choice, dict(decision, step="selection step 4", reason=reason)


def descriptive(means: np.ndarray, reference: np.ndarray, rows: list[dict], tau: float | None, eps: float) -> dict:
    _, margins = best_roots(means)
    picks0 = gated_picks(means, 0)
    overrides = picks0 != 0
    bins = (0, .001, .01, .03, .1, math.inf)
    histogram = []
    for lo, hi in zip(bins, bins[1:]):
        bucket = (margins >= lo) & (margins < hi)
        n_over = int((bucket & overrides).sum())
        histogram.append({"bucket": f"[{lo:g},{hi:g})", "count": int(bucket.sum()),
                          "share_of_pairs": float(bucket.mean()) if bucket.size else None,
                          "n_over": n_over, "share_of_overrides": n_over / int(overrides.sum()) if overrides.any() else None})
    def agreement(picks):
        return float(np.mean([np.mean(picks[:, a] == picks[:, b]) for a in range(4) for b in range(a + 1, 4)])) if len(picks) else None
    teams = {}
    if tau is not None:
        for game in GAMES:
            mask = np.asarray([row["game"] == game for row in rows], dtype=bool)
            if mask.any():
                teams[game] = metrics(means[mask], reference[mask], tau, eps, None)
    return {"used_by_rule": False, "tau0_override_rate": float(overrides.mean()) if overrides.size else None,
            "T0061_override_rate_for_context": .73, "margin_buckets": histogram,
            "repeat_pair_agreement": {"tau0": agreement(picks0), "frozen_tau": tau,
                                      "at_frozen_tau": agreement(gated_picks(means, tau)) if tau is not None else None},
            "per_team_count_at_frozen_tau": teams}


def integrity_summary(receipt: dict) -> dict:
    summary = {k: v for k, v in receipt.items() if k != "commands"}
    summary["commands"] = []
    for command in receipt.get("commands", []):
        # The tree file and INTEGRITY.json retain the large raw outputs; avoid duplicating them again.
        entry = dict(command)
        if entry.get("kind") == "diagnostic":
            entry["output_line_count"] = len(entry.pop("stdout_lines", []))
        summary["commands"].append(entry)
    return summary


def analyze(rows: list[dict], receipt: dict, trees_hash: str) -> dict:
    split_rows = {split: [r for r in rows if r["split"] == split] for split in ("selection", "validation")}
    eps = max((t["max_abs_f64"] for row in split_rows["selection"] for t in row["trees"]), default=0.0)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    resamples, bootstrap_hashes = {}, {}
    for split, states in split_rows.items():
        n = len(states)
        resamples[split] = rng.integers(0, n, size=(BOOTSTRAPS, n), dtype=np.int64) if n else None
        bootstrap_hashes[split] = hashlib.sha256(resamples[split].astype("<i8").tobytes()).hexdigest() if n else None
    data = {name: {split: arrays(states, *config) for split, states in split_rows.items()}
            for name, config in CONFIGS.items()}
    # Freeze the selection choice before computing any validation quantities.
    tables = {name: {"selection": [metrics(*data[name]["selection"], tau, eps, resamples["selection"]) for tau in GRID]}
              for name in CONFIGS}
    choice, decision = choose({name: tables[name]["selection"] for name in CONFIGS})
    for name in CONFIGS:
        tables[name]["validation"] = [metrics(*data[name]["validation"], tau, eps, resamples["validation"]) for tau in GRID]
    validation = {"looked_at": choice is not None, "passed": False, "reason": "no supported selection choice"}
    if choice is not None:
        row = next(r for r in tables[choice["breadth"]]["validation"] if r["tau"] == choice["tau"])
        lower = row["C_ci95"][0] if row["C_ci95"] is not None else None
        checks = {"n_over_ge_10": row["n_over"] >= 10, "C_lower_gt_0.5": lower is not None and lower > .5,
                  "G_gt_0": row["G"] is not None and row["G"] > 0}
        validation = {"looked_at": True, "passed": all(checks.values()), "checks": checks, "metrics": row,
                      "reason": "all frozen validation criteria pass" if all(checks.values()) else "frozen validation criteria fail; no retuning"}
    expected = {(g, seat, turn) for g, teams in GAMES.items() for seat in range(teams) for turn in range(4, 401, 4)}
    actual = {(r["game"], r["seat"], r["turn"]) for r in rows}
    kinds = [r.get("kind") for r in receipt.get("commands", []) if r.get("passed") is True]
    integrity_ok = (receipt.get("passed") is True and all(r.get("passed") is True for r in receipt.get("commands", [])) and
                    all(kinds.count(kind) >= count for kind, count in (("gate", 1), ("isolation", 6),
                                                                     ("parity_wide", 3), ("parity_narrow", 3))))
    hash_ok = receipt.get("trees_sha256") == trees_hash
    coverage = {"expected_states": len(expected), "actual_states": len(actual), "missing": len(expected - actual),
                "extra": len(actual - expected), "complete": actual == expected and not receipt.get("smoke", False)}
    if not integrity_ok or not hash_ok:
        frozen = {"blocked": "integrity stop: failed/missing checks or tree SHA-256 mismatch"}
        final_step = "integrity stop"
    elif not coverage["complete"]:
        frozen = {"blocked": "incomplete pre-registered state corpus (smoke/subset results cannot freeze a gate)"}
        final_step = "states requirement"
    elif choice is None:
        frozen = {"blocked": decision["reason"]}
        final_step = decision["step"]
    elif not validation["passed"]:
        frozen = {"blocked": validation["reason"]}
        final_step = "validation"
    else:
        frozen = choice
        final_step = "selection step 4, followed by validation pass"
    all_data = {name: arrays(rows, *config) for name, config in CONFIGS.items()}
    descriptions = {name: {"all": descriptive(*all_data[name], rows, choice["tau"] if choice else None, eps),
                           **{split: descriptive(*data[name][split], states, choice["tau"] if choice else None, eps)
                              for split, states in split_rows.items()}} for name in CONFIGS}
    return {"schema": 1, "rule": "docs/AGENT_V2_SEARCH_GATE.md section 1 (unchanged)",
            "scope": "Numerical stability on fixed trees; says nothing about winning games.", "frozen": frozen,
            "deciding_step": final_step, "selection_decision": decision, "selection_choice": choice,
            "configurations": {name: {"candidates": k, "samples": s} for name, (k, s) in CONFIGS.items()},
            "eps": eps, "twice_eps": 2 * eps, "grid": GRID, "states": {s: len(r) for s, r in split_rows.items()},
            "coverage": coverage, "bootstrap": {"resamples": BOOTSTRAPS, "generator": "numpy PCG64", "seed": BOOTSTRAP_SEED,
                "unit": "state, retaining all four repeats", "interval": "percentile [2.5, 97.5]",
                "order": "selection then validation; same state draws for all configs and thresholds within each split",
                "undefined_C": "omit resamples with no overrides from C percentiles; count reported per row",
                "index_sha256_le_i64": bootstrap_hashes},
            "tables": tables, "validation": validation, "descriptive_not_used_by_rule": descriptions,
            "integrity_passed": integrity_ok and hash_ok, "integrity": integrity_summary(receipt), "trees_sha256": trees_hash}


def fmt(value, digits=6) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def interval(row: dict, key: str) -> str:
    ci = row[key + "_ci95"]
    return fmt(row[key]) + (f" [{fmt(ci[0])}, {fmt(ci[1])}]" if ci is not None else " [undefined]")


def markdown(report: dict) -> str:
    frozen, choice = report["frozen"], report["selection_choice"]
    lines = ["# agent_v2 offline search gate diagnostics", "", "**Do not edit.** Generated by `tools/screens/search_gate_offline/analyze.py`.", "",
             "This is numerical stability on fixed trees. It says nothing about winning games.", ""]
    if "blocked" in frozen:
        lines += [f"**Blocked:** {frozen['blocked']}"]
    else:
        lines += [f"**Frozen:** {frozen['breadth']} ({frozen['candidates']} candidates × {frozen['samples']} scenarios), tau = {fmt(frozen['tau'], 3)}."]
    lines += [f"Deciding rule step: {report['deciding_step']}.", "",
              f"Selection eps = {report['eps']:.9g}; 2 eps = {report['twice_eps']:.9g} (all selection trees, including reference).",
              f"States: selection {report['states']['selection']}, validation {report['states']['validation']}; full corpus: {report['coverage']['complete']}.",
              f"Selection decision: {report['selection_decision']['step']}: {report['selection_decision']['reason']}.",
              "tau_c: " + ", ".join(f"{name} = {fmt(tau, 3)}" for name, tau in report["selection_decision"]["tau_c"].items()) + ".", "",
              "Intervals use 2,000 state bootstrap draws, with all four repeats, numpy PCG64 and the registered SHA-256 seed. "
              "C is undefined without overrides; bootstrap resamples without overrides are omitted from C percentiles (counts in JSON).", ""]
    for name, splits in report["tables"].items():
        for split, rows in splits.items():
            lines += [f"## {name}: {split}", "", "| tau | n_over | Override rate | C [95% CI] | G [95% CI] | Eligible |",
                      "| ---: | ---: | ---: | ---: | ---: | :---: |"]
            for row in rows:
                lines.append(f"| {fmt(row['tau'], 3)} | {row['n_over']} | {fmt(row['override_rate'])} | {interval(row, 'C')} | {interval(row, 'G')} | {'yes' if row['eligible'] else 'no'} |")
            lines += [""]
    lines += ["Eligibility shown on validation uses the selection criterion for display only. Validation grid rows do not select or retune anything.", "",
              "## Validation of the selection choice", ""]
    validation = report["validation"]
    if choice:
        row = validation["metrics"]
        lines += [f"{'PASS' if validation['passed'] else 'FAIL'} at {choice['breadth']}, tau {fmt(choice['tau'], 3)}: "
                  f"n_over = {row['n_over']} (needs >= 10), C = {interval(row, 'C')} (lower needs > 0.5), "
                  f"G = {fmt(row['G'])} (needs > 0). No retuning."]
    else:
        lines += ["Not evaluated: no supported selection choice."]
    lines += ["", "## Descriptive only — not used by the rule", "",
              "T0061's 73% override rate is historical context from a different carrier/state corpus; this is not a matched comparison.", ""]
    if choice and "blocked" in frozen:
        lines += ["No gate is frozen. The 'frozen tau' descriptive columns below use the selection choice provisionally; they do not authorize a host screen.", ""]
    for name, splits in report["descriptive_not_used_by_rule"].items():
        for split, desc in splits.items():
            agreement = desc["repeat_pair_agreement"]
            lines += [f"### {name}: {split}", "", f"Override rate at tau 0: {fmt(desc['tau0_override_rate'])} (T0061: 0.730000).",
                      f"Repeat-pair agreement (including actor keeps): tau 0 {fmt(agreement['tau0'])}; "
                      f"frozen tau {fmt(agreement['frozen_tau'], 3)}: {fmt(agreement['at_frozen_tau'])}.", "",
                      "| Best-minus-actor margin | Pairs | Share of pairs | Overrides at tau 0 | Share of overrides |",
                      "| :--- | ---: | ---: | ---: | ---: |"]
            for bucket in desc["margin_buckets"]:
                lines.append(f"| {bucket['bucket']} | {bucket['count']} | {fmt(bucket['share_of_pairs'])} | {bucket['n_over']} | {fmt(bucket['share_of_overrides'])} |")
            lines += [""]
        lines += [f"### {name}: per team count at frozen tau", "", "| Game | States | n_over | Override rate | C | G |",
                  "| :--- | ---: | ---: | ---: | ---: | ---: |"]
        teams = splits["all"]["per_team_count_at_frozen_tau"]
        for game, row in teams.items():
            lines.append(f"| {game} | {row['states']} | {row['n_over']} | {fmt(row['override_rate'])} | {fmt(row['C'])} | {fmt(row['G'])} |")
        if not teams:
            lines.append("| No supported selection tau | — | — | — | — | — |")
        lines += [""]
    lines += ["## Integrity", "", f"Overall: {'PASS' if report['integrity_passed'] else 'FAIL'}; tree SHA-256 `{report['trees_sha256']}`.", "",
              "| Check | Result | Output / comparison |", "| :--- | :---: | :--- |"]
    for command in report["integrity"].get("commands", []):
        if command.get("kind") in ("gate", "isolation", "parity_wide", "parity_narrow"):
            output = command.get("failure", command.get("comparison", " ".join(command.get("stdout_lines", []))))
            lines.append(f"| {command['kind']} | {'PASS' if command.get('passed') else 'FAIL'} | {output.replace('|', '/')} |")
    lines += ["", "Commands, literal outputs, compiler version, flags, git HEAD and input hashes are in INTEGRITY.json. "
              "Simulator/engine step parity belongs to tests/test_agent_v2.py with an arena checkout; this diagnostic does not claim to re-run it.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trees", type=Path, default=OUT / "TREES.jsonl.gz")
    ap.add_argument("--integrity", type=Path, help="default: INTEGRITY.json beside --trees")
    ap.add_argument("--out", type=Path, default=OUT, help="directory for generated DIAGNOSTICS.json and .md")
    args = ap.parse_args()
    integrity = args.integrity or args.trees.with_name("INTEGRITY.json")
    rows = read_states(args.trees)
    receipt = json.loads(integrity.read_text(encoding="utf-8"))
    report = analyze(rows, receipt, sha256(args.trees))
    report["integrity_sha256"] = sha256(integrity)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "DIAGNOSTICS.json").write_text(json_text(report) + "\n", encoding="utf-8")
    (args.out / "DIAGNOSTICS.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"frozen": report["frozen"], "states": report["states"], "integrity_passed": report["integrity_passed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
