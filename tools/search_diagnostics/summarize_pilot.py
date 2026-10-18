"""T0086 summary: REPORT.json and REPORT.md from the committed receipts (task T0092).

    python tools/search_diagnostics/summarize_pilot.py [--dir DIR] [--audit EVIDENCE_AUDIT.json] [--check]

Reads, under --dir (default reports/host/search_branch_diagnostics): ANALYSIS_PRIMARY.json, ANALYSIS_EXTENDED.json,
PARITY.json, P2_EVAL.json, P3_EVAL.json, ESTIMATE.json, sources/screen.json and sources/RUN.json; and T0084's
EVIDENCE_AUDIT.json for the search completion of the T0081 screen. A missing input refuses the run.

Every figure is read or computed from these files. Uncertainty is analyze.aggregate's: source maps are the clusters,
team counts weigh equally, intervals are normal 95%, and a count with one map leaves the estimate without one. A
missing metric is unavailable, never zero. Figures no committed receipt records (the number of branches played) are
marked unavailable. The gates follow GATE_RULES. T0086 named the gates but registered no thresholds; the rules were
written by T0092 after the collection, as the reading of T0086's text (an opportunity has to survive selection on
some repeats and evaluation on disjoint ones), and are not a pre-registration.

REPORT.md is rendered from REPORT.json alone. --check exits 1 when the committed pair differs from the receipts.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.search_diagnostics.analyze import aggregate  # noqa: E402

SCHEMA = "ctf-host-search-branch-diagnostics-report-2"
DIR = REPO / "reports" / "host" / "search_branch_diagnostics"
AUDIT = REPO / "reports" / "host" / "agent_v2_p3_search" / "EVIDENCE_AUDIT.json"
INPUTS = {"primary": "ANALYSIS_PRIMARY.json", "extended": "ANALYSIS_EXTENDED.json", "parity": "PARITY.json",
          "p2": "P2_EVAL.json", "p3": "P3_EVAL.json", "estimate": "ESTIMATE.json",
          "screen": "sources/screen.json", "run": "sources/RUN.json"}
METRICS = ("in_sample_best_gain", "cross_fit_gain", "v_gate_gain", "v_override", "v_spearman", "gain_repeat_sd")
EXTENDED_METRICS = ("in_sample_best_gain", "cross_fit_gain", "v_gate_gain", "v_gate24_gain")
SIGNED = ("cross_fit_gain", "v_gate_gain", "v_gate24_gain", "v_spearman")   # the others are >= 0 by construction
GATE_RULES = {
    "A": "permitted when cross_fit_gain over all analysed primary roots has a 95% interval above zero and v_spearman "
         "over the same roots has no interval above zero",
    "B": "applies when that cross_fit_gain is not shown above zero (its interval includes zero, or it has none); a "
         "finite sample that does not show an opportunity does not show its absence",
    "C": "triggered when the parity receipt is not passed, reports a difference or has an event short of its minimum",
    "D": "permitted when a search arm of the T0081 screen completed search on under half of its turns at some team "
         "count with complete counters (EVIDENCE_AUDIT search_telemetry)",
}
FOLLOW_UP = {"A": "T0088", "D": "T0087"}
LIMITS = [
    "Parity covers only the team counts and event coverage listed; other counts and rarer event combinations are "
    "untested.",
    "P fidelity is measured on factual recorded states only; it says nothing about P on branched (counterfactual) "
    "states.",
    "The contact and override strata are oversampled tactical turns and represent only themselves; the all-roots pool "
    "mixes strata and is not a population estimate.",
    "Roots refused on replay are excluded, not imputed, so the analysed roots are not a random subset of the "
    "registered ones (see the exclusions by opponent and the analysed roots per count).",
    "A cross-fit gain that is not shown does not show zero opportunity, and a weak or unshown V ranking is not a "
    "unique cause of the T0081 search deficit.",
]
DIVERGED = re.compile(r"seat \d+ \((?P<bot>[^)]+)\) diverged from the record")


def load(directory: Path, audit: Path) -> dict:
    paths = {k: directory / v for k, v in INPUTS.items()} | {"audit": audit}
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise SystemExit("missing inputs, nothing written:\n  " + "\n  ".join(missing))
    return {k: json.loads(p.read_text(encoding="utf-8")) for k, p in paths.items()}


# ---- estimates --------------------------------------------------------------------------------------------------

def compact(a: dict) -> dict:
    """The fields of an analyze.aggregate result the report uses."""
    if a.get("status") != "computed":
        return {"status": a.get("status", "unavailable"), "missing_counts": a.get("missing_counts", [])}
    return {"status": "computed", "counts": a["counts"], "mean": a["mean_equal_count_weight"], "ci95": a["ci95_normal"],
            "se_status": a["se_status"], "missing_counts": a["missing_counts"], "missing_roots": a["missing_roots"]}


def verdict(e: dict) -> str:
    if e["status"] != "computed":
        return "unavailable"
    if e["ci95"] is None:
        return "not shown (no interval)"
    lo, hi = e["ci95"]
    return "above zero" if lo > 0 else "below zero" if hi < 0 else "not shown"


def estimates(rows: list[dict], metrics, counts: list[int]) -> dict:
    return {m: compact(aggregate(rows, m, counts)) for m in metrics}


def with_verdicts(block: dict) -> dict:
    return {m: e | {"verdict": verdict(e)} if m in SIGNED else e for m, e in block.items()}


# ---- sections ---------------------------------------------------------------------------------------------------

def exclusions(analysis: dict) -> dict:
    by = Counter()
    for reason, n in analysis["excluded"].items():
        m = DIVERGED.match(reason)
        by[f"{m['bot']} diverged from its recorded game on replay" if m else reason] += n
    return dict(sorted(by.items(), key=lambda kv: -kv[1]))


def roots_section(inp: dict, counts: list[int]) -> dict:
    est, pri, ext = inp["estimate"], inp["primary"], inp["extended"]
    design = est["design"]
    analysed = Counter(r["teams"] for r in pri["per_root"])
    return {
        "primary": {"registered": est["roots"], "extracted": pri["roots"], "registered_not_extracted": est["roots"] - pri["roots"],
                    "analysed": pri["analysed"], "excluded": exclusions(pri),
                    "per_count": {str(c): {"registered": est["per_count"][str(c)]["roots"], "analysed": analysed.get(c, 0)}
                                  for c in counts},
                    "complete_cells": sum(r["candidates"] * r["repeats"] for r in pri["per_root"])},
        "extended": {"registered": design["subset_maps_per_count"] * design["subset_roots_per_map"] * len(counts),
                     "extracted": ext["roots"], "analysed": ext["analysed"], "excluded": exclusions(ext),
                     "complete_cells": sum(r["candidates"] * r["repeats"] for r in ext["per_root"])},
        "episode_caps": {"primary": est["primary_episodes"], "extended": est["subset_episodes"], "total": est["episodes"],
                         "source_games": est["source_games"]},
        "branches_played": "unavailable: no committed receipt counts the branches played, only the complete ones",
    }


def sources_section(inp: dict) -> dict:
    screen, run = inp["screen"], inp["run"]
    (arm, a), = screen["arms"].items()
    return {"seed": screen["seed"], "fixtures": screen["fixtures"], "arm": arm, "host": run["host"], "os": run["os"],
            "workers": run["effective"]["workers"], "limits_ms": [run["config"]["first_ms"], run["config"]["turn_ms"]],
            "opponents": run["config"]["reference"], "map_splits": run["config"]["map_splits"],
            **{k: a[k] for k in ("games", "faults", "late_replies", "opponent_faults", "opponent_late", "max_ms", "p99_ms_worst")},
            "wall_seconds": run["wall_seconds"]["total"]}


def parity_section(par: dict) -> dict:
    return {k: par[k] for k in ("status", "steps_compared", "team_counts", "steps_per_count", "differences", "coverage",
                                "minimums", "short_of_minimum", "scope")}


def p_fidelity(p2: dict, p3: dict) -> dict:
    out = {"scope": p3["scope"], "games": p3["games"], "events": {}}
    for event in p3["event_definitions"]:
        both = [c for c in p3["by_count"] if p2["by_count"].get(c, {}).get(event, {}).get("nll") is not None
                and p3["by_count"][c][event].get("nll") is not None]
        row = {"counts": len(both), "missing_counts": sorted(set(p3["by_count"]) - set(both), key=int),
               "labels": sum(p3["by_count"][c][event]["count"] for c in both)}
        for name, p in (("p2", p2), ("p3", p3)):
            for k in ("nll", "top1_accuracy"):
                vals = [p["by_count"][c][event].get(k) for c in both]
                row[f"{name}_{k}"] = sum(vals) / len(vals) if both and None not in vals else None
        row["p3_minus_p2_nll"] = None if row["p3_nll"] is None else row["p3_nll"] - row["p2_nll"]
        out["events"][event] = row
    return out


def completion_section(audit: dict) -> dict:
    tel = audit.get("search_telemetry", {})
    if tel.get("status") != "computed":
        return {"status": "unavailable", "reason": tel.get("reason", "no search_telemetry in the audit")}
    arms = {}
    for arm, per in tel["per_arm"].items():
        if not any(d["search_used"] + d["search_fallback"] for d in per.values()):
            continue
        arms[arm] = {"below_half": {c: d["completion"] for c, d in per.items()
                                    if d["complete"] and d["completion"] is not None and d["completion"] < 0.5},
                     "incomplete_counts": [c for c, d in per.items() if not d["complete"]]}
    return {"status": "computed", "schema": audit["schema"], "arms": arms}


def gates(pooled_all: dict, parity: dict, completion: dict) -> dict:
    cf, vs = pooled_all["cross_fit_gain"], pooled_all["v_spearman"]
    shown = verdict(cf) == "above zero"
    v_ranks = verdict(vs) == "above zero"
    c_trig = parity["status"] != "passed" or bool(parity["differences"]) or bool(parity["short_of_minimum"])
    d_counts = {a: sorted(v["below_half"], key=int) for a, v in completion.get("arms", {}).items() if v["below_half"]}
    out = {
        "A": {"decision": "permitted" if shown and not v_ranks else "not permitted",
              "inputs": {"cross_fit_gain_all_roots": cf, "v_spearman_all_roots": vs}},
        "B": {"decision": "applies" if not shown else "does not apply",
              "inputs": {"cross_fit_gain_all_roots": cf},
              "reading": "uncertainty: the interval includes zero or is unavailable" if not shown else None},
        "C": {"decision": "triggered" if c_trig else "not triggered",
              "inputs": {k: parity[k] for k in ("status", "differences", "short_of_minimum", "steps_compared")}},
        "D": {"decision": ("permitted" if d_counts else "not permitted") if completion["status"] == "computed" else "unavailable",
              "inputs": {"counts_below_half": d_counts}},
    }
    for k, g in out.items():
        g["rule"] = GATE_RULES[k]
        g["follow_up"] = FOLLOW_UP.get(k) if g["decision"] == "permitted" else None
    return out


def summarize(inp: dict) -> dict:
    counts = inp["estimate"]["design"]["counts"]
    rows = inp["primary"]["per_root"]
    strata = sorted({r["stratum"] for r in rows})
    pooled = {s: with_verdicts(estimates([r for r in rows if r["stratum"] == s], METRICS, counts)) for s in strata}
    pooled["all"] = with_verdicts(estimates(rows, METRICS, counts))
    groups = {g: with_verdicts({m: compact(v[m]) for m in METRICS}) for g, v in sorted(inp["primary"]["groups"].items())}
    parity = parity_section(inp["parity"])
    completion = completion_section(inp["audit"])
    return {"schema": SCHEMA, "task": "T0086", "corrected_by": "T0092", "counts": counts,
            "analysis": {k: inp["primary"][k] for k in ("schema", "collection", "tau")},
            "sources": sources_section(inp), "roots": roots_section(inp, counts), "parity": parity,
            "p_fidelity": p_fidelity(inp["p2"], inp["p3"]),
            "primary": {"by_split_and_stratum": groups, "by_stratum_all_splits": pooled},
            "extended": with_verdicts(estimates(inp["extended"]["per_root"], EXTENDED_METRICS, counts)),
            "t0081_search_completion": completion,
            "gates": gates(pooled["all"], parity, completion), "limits": LIMITS}


# ---- rendering --------------------------------------------------------------------------------------------------

def fmt(e: dict) -> str:
    if e["status"] != "computed":
        return "n/a"
    ci = "no interval" if e["ci95"] is None else f"[{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]"
    shown = f", {e['verdict']}" if e.get("verdict") in ("above zero", "below zero", "not shown") else ""
    return f"{e['mean']:+.4f} {ci}{shown}"


def num(x, f: str) -> str:
    return "n/a" if x is None else format(x, f)


def render(doc: dict) -> str:
    s, r, par, g = doc["sources"], doc["roots"], doc["parity"], doc["gates"]
    L = ["# Bounded branch diagnostics (T0086)", "",
         "Generated by `tools/search_diagnostics/summarize_pilot.py` (task T0092) from the committed receipts in this "
         "directory and `reports/host/agent_v2_p3_search/EVIDENCE_AUDIT.json`. Do not edit by hand. The hand-written "
         "report of commit 5c20cc4 is withdrawn; see [CORRECTION.md](CORRECTION.md).", "",
         "Means weigh team counts equally and source maps are the clusters (`analyze.aggregate`). Intervals are 95% "
         "normal; a team count with a single map leaves an estimate without one. \"Not shown\" is not evidence of "
         "zero. The gate rules were written after the collection (see Decision gates).", "",
         "## Source games", "",
         f"Seed {s['seed']}: {s['games']} games of `{s['arm']}` on {s['host']} ({s['os']}), {s['workers']} workers, "
         f"limits {s['limits_ms'][0]}/{s['limits_ms'][1]} ms, opponents {', '.join(s['opponents'])}. Candidate "
         f"faults {s['faults']}, late replies {s['late_replies']}, opponent faults {s['opponent_faults']}, opponent "
         f"late {s['opponent_late']}; max reply {s['max_ms']} ms; run wall time {s['wall_seconds']} s. Map splits: "
         + "; ".join(f"{k} {v}" for k, v in s["map_splits"].items()) + ".", "",
         "## Roots and branches", ""]
    for name in ("primary", "extended"):
        x = r[name]
        ex = "; ".join(f"{k}: {v}" for k, v in x["excluded"].items()) or "none"
        L.append(f"- {name}: {x['registered']} registered, {x['extracted']} extracted, {x['analysed']} analysed, "
                 f"{x['complete_cells']} complete candidate x repeat cells in the analysed roots. Excluded: {ex}.")
    caps = r["episode_caps"]
    L += ["- The extended set reuses the verified primary branches of its roots, so its cells overlap the primary ones.",
          f"- Caps: {caps['primary']} primary and {caps['extended']} extended branch episodes ({caps['total']} total), "
          f"{caps['source_games']} source games. Branches played: {r['branches_played']}.", "",
          "| Teams | " + " | ".join(r["primary"]["per_count"]) + " |",
          "|---|" + "---:|" * len(r["primary"]["per_count"]),
          "| analysed / registered roots | " + " | ".join(f"{v['analysed']}/{v['registered']}"
                                                          for v in r["primary"]["per_count"].values()) + " |", "",
          "## Simulator parity", "",
          f"Status {par['status']}: {par['steps_compared']} steps ({par['steps_per_count']} per count at team counts "
          f"{', '.join(map(str, par['team_counts']))}), {par['differences']} differences. Event coverage: "
          + ", ".join(f"{k} {v}" for k, v in par["coverage"].items())
          + f". Short of minimum: {', '.join(par['short_of_minimum']) or 'none'}. Scope: {par['scope']}.", "",
          "## Opponent prediction on factual states", "",
          f"{doc['p_fidelity']['games']} games; {doc['p_fidelity']['scope']}. Equal-count means over the counts "
          "where both models have a value; no interval (the receipts have no per-map breakdown).", "",
          "| Event | Counts | Labels | P2 NLL | P3 NLL | P3 - P2 NLL | P2 top-1 | P3 top-1 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for e, x in doc["p_fidelity"]["events"].items():
        L.append(f"| {e} | {x['counts']} | {x['labels']} | {num(x['p2_nll'], '.4f')} | {num(x['p3_nll'], '.4f')} | "
                 f"{num(x['p3_minus_p2_nll'], '+.4f')} | {num(x['p2_top1_accuracy'], '.4f')} | {num(x['p3_top1_accuracy'], '.4f')} |")
    head = "| Group | " + " | ".join(METRICS) + " |"
    rule = "|---|" + "---|" * len(METRICS)
    L += ["", "## Primary set: all splits, by stratum", "",
          "In-sample best gain is a maximum over noisy candidates, biased upward and never a demonstrated gain. "
          "Cross-fit gain selects on some repeats and evaluates on disjoint ones. v_gate_gain is the frozen gate's pick.",
          "", head, rule]
    for name, block in doc["primary"]["by_stratum_all_splits"].items():
        L.append(f"| {name} | " + " | ".join(fmt(block[m]) for m in METRICS) + " |")
    L += ["", "## Primary set: by split and stratum (as analysed)", "", head, rule]
    for name, block in doc["primary"]["by_split_and_stratum"].items():
        L.append(f"| {name} | " + " | ".join(fmt(block[m]) for m in METRICS) + " |")
    L += ["", "## Extended set (24 candidates)", "", "| Metric | Estimate | Counts | Missing counts |", "|---|---|---:|---|"]
    for m, e in doc["extended"].items():
        L.append(f"| {m} | {fmt(e)} | {e.get('counts', 0)} | {', '.join(map(str, e['missing_counts'])) or '-'} |")
    comp = doc["t0081_search_completion"]
    L += ["", "## Search completion in the T0081 screen", ""]
    if comp["status"] != "computed":
        L.append(f"Unavailable: {comp['reason']}.")
    for arm, x in comp.get("arms", {}).items():
        below = ", ".join(f"n{c} {100 * v:.1f}%" for c, v in sorted(x["below_half"].items(), key=lambda kv: int(kv[0])))
        L.append(f"- `{arm}`: completion under half at {below or 'no count'}"
                 + (f"; counts without complete counters: {', '.join(x['incomplete_counts'])}" if x["incomplete_counts"] else "") + ".")
    L += ["", "## Decision gates", "", "| Gate | Decision | Rule | Inputs | Follow-up |", "|---|---|---|---|---|"]
    for k, x in g.items():
        if k in ("A", "B"):
            inputs = "; ".join(f"{n.replace('_all_roots', '')} {fmt(v)}" for n, v in x["inputs"].items())
            inputs += f" ({x['reading']})" if x.get("reading") else ""
        elif k == "C":
            inputs = "; ".join(f"{n} {v if v not in ([], {}) else 'none'}" for n, v in x["inputs"].items())
        else:
            inputs = "; ".join(f"`{a}`: n{', n'.join(c)}" for a, c in x["inputs"]["counts_below_half"].items()) or "none"
        L.append(f"| {k} | **{x['decision']}** | {x['rule']} | {inputs} | {x['follow_up'] or '-'} |")
    L += ["", "## Limits", ""] + [f"- {t}" for t in doc["limits"]]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DIR)
    ap.add_argument("--audit", type=Path, default=AUDIT)
    ap.add_argument("--check", action="store_true", help="compare with the committed REPORT.json/.md; write nothing")
    a = ap.parse_args(argv)
    doc = summarize(load(a.dir, a.audit))
    text_json, text_md = json.dumps(doc, indent=1) + "\n", render(doc)
    if a.check:
        stale = [n for n, t in (("REPORT.json", text_json), ("REPORT.md", text_md))
                 if not (a.dir / n).is_file() or (a.dir / n).read_text(encoding="utf-8") != t]
        if stale:
            sys.stderr.write(f"stale: {', '.join(stale)}; run python tools/search_diagnostics/summarize_pilot.py\n")
            return 1
        return 0
    (a.dir / "REPORT.json").write_text(text_json, encoding="utf-8")
    (a.dir / "REPORT.md").write_text(text_md, encoding="utf-8")
    print(f"[summary] wrote {a.dir / 'REPORT.json'} and REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
