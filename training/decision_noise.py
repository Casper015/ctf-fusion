"""Noise, coverage and power diagnostics of a decision dataset (task T0034, docs/RL_PLAN.md section 3).

    python -m training.decision_noise --data data/decision/t0010_p2 --out reports/models/decision/noise

Reported per split and per team count, never turned into an automatic kill rule:
  coverage   roots, effective (independent) maps, candidates per root (actual K), supported candidates, label
             availability (candidates without any continuation), changeable roots (incumbent labeled and at least
             one supported, labeled alternative), canonicalization flags, effective duplicates, targets outside the
             representation, feature collisions (distinct canonical orders with identical scorer features);
  noise      over one population, the noise roots (two or more noise candidates: the incumbent and supported
             candidates with an outcome): between-action variance of candidate means inside a root, raw and with
             the expected within-action share removed; within-action repeat variance (pooled sample variance of
             repeats of the *same* candidate), only where two or more outcomes exist; the paired candidate-minus-
             incumbent variance over shared repeat ids, with the covariance it implies. NOISE_ASSUMPTIONS states
             what each needs. The within-action variance of every candidate is reported apart and never corrects
             anything. Spread of effects across *different* roots is never used as continuation noise;
  faults     per phase (prefix replay, discarded root-turn replies, live continuation) and failed repeats;
  replay     the original game's outcome against the incumbent's replayed continuations: a memory/replay
             diagnostic (bots restart from the recorded prefix), reported apart from repeat noise;
  power      a planning approximation from map-level variance of an always-switch rule on train/val:
             MDE(M, f) = 2.8 * sqrt(f * s2_map / M) for M independent maps and switch share f.

The test split is read only with --selection, after training.evaluate_decisions verified the frozen selection.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from training import decision_dataset as D
from training.evaluate_decisions import count_weights, roots_of


NOISE_ASSUMPTIONS = (
    "Every statistic below is computed over one population: the noise roots, i.e. roots with at least two noise "
    "candidates (the incumbent and supported candidates with an outcome). Unsupported candidates, candidates without an "
    "outcome and roots with fewer than two noise candidates enter none of them, so adding such data changes nothing.",
    "between_action_variance_net = raw minus within-action variance x the root's mean 1/n, averaged over roots. It "
    "assumes one within-action variance for every noise candidate, independent repeats inside a candidate, and no "
    "covariance between the repeat noise of different candidates at a root; positive covariance would make the "
    "correction too large and the net value too small.",
    "Repeats are paired by repeat id (the same column of cand_outcome). The collector has no controlled seed "
    "(decision_branch_collect.RNG_NOTE): matching repeat ids neither create nor prove correlation, and a small implied "
    "covariance is consistent with, but does not prove, independent identically distributed repeats.",
)


def noise_population(d: dict) -> list[tuple[int, np.ndarray]]:
    """(root, its noise candidates) for every root with at least two: the incumbent and supported candidates, each
    with at least one outcome. The between-action statistic and every correction of it use exactly these."""
    n = d["cand_n"].astype(np.int64)
    out = []
    for r, idx in roots_of(d):
        c = idx[(d["cand_support"][idx] | (idx == d["root_incumbent"][r])) & (n[idx] > 0)]
        if len(c) >= 2:
            out.append((r, c))
    return out


def pooled_variance(outcomes: np.ndarray) -> tuple[float | None, int]:
    """Pooled sample variance of the rows of (k, R) outcomes (NaN = missing), rows with two or more outcomes."""
    num, dof = 0.0, 0
    for o in outcomes:
        o = o[np.isfinite(o)]
        if len(o) >= 2:
            num += (len(o) - 1) * np.var(o, ddof=1)
            dof += len(o) - 1
    return (num / dof if dof else None), dof


def paired_gaps(d: dict, pop) -> dict:
    """Candidate-minus-incumbent differences of outcomes with the same repeat id, at the noise roots."""
    out = d["cand_outcome"]
    num = num_c = num_i = 0.0
    dof = pairs = 0
    for r, c in pop:
        inc = int(d["root_incumbent"][r])
        if inc not in c:
            continue
        for a in c:
            if a == inc:
                continue
            both = np.isfinite(out[a]) & np.isfinite(out[inc])
            m = int(both.sum())
            if m < 2:
                continue
            gap = out[a][both] - out[inc][both]
            num += (m - 1) * np.var(gap, ddof=1)
            num_c += (m - 1) * np.var(out[a][both], ddof=1)
            num_i += (m - 1) * np.var(out[inc][both], ddof=1)
            dof += m - 1
            pairs += 1
    if not dof:
        return {"paired_gap_variance": None, "pairs": pairs,
                "note": "unavailable: no candidate shares two or more repeat ids with an outcome with its incumbent"}
    v, vc, vi = num / dof, num_c / dof, num_i / dof
    return {"paired_gap_variance": v, "dof": dof, "pairs": pairs, "candidate_variance": vc, "incumbent_variance": vi,
            "implied_covariance": (vc + vi - v) / 2,
            "note": "pooled over candidate/incumbent pairs on their shared repeat ids; independent repeats would give "
                    "paired_gap_variance = candidate_variance + incumbent_variance (implied covariance 0)"}


def repeat_stats(d: dict) -> dict:
    n = d["cand_n"].astype(np.int64)
    state = d["cand_repeat_state"]
    out = {"repeats": {str(k): int(v) for k, v in sorted(Counter(n.tolist()).items())},
           "failed_repeats": int((state == D.REPEAT_FAILED).sum()), "assumptions": list(NOISE_ASSUMPTIONS)}
    pop = noise_population(d)
    cands = np.concatenate([c for _, c in pop]) if pop else np.zeros(0, np.int64)
    within, dof = pooled_variance(d["cand_outcome"][cands])
    out["noise_roots"] = len(pop)
    out["noise_candidates"] = int(len(cands))
    out["within_action_variance"] = within
    out["within_action_dof"] = dof
    if within is None:
        out["within_action_note"] = "unavailable: no noise candidate has two repeats with an outcome"
    every, every_dof = pooled_variance(d["cand_outcome"][n >= 2])
    out["within_action_variance_all_candidates"] = every
    out["within_action_all_note"] = ("every candidate with two or more outcomes, supported or not, at any root: "
                                     "reported for comparison, never used to correct the between-action variance")
    raw = [np.var(d["cand_y"][c], ddof=1) for _, c in pop]
    inv_n = [np.mean(1.0 / n[c]) for _, c in pop]
    out["between_action_variance_raw"] = float(np.mean(raw)) if raw else None
    out["between_roots"] = len(raw)
    if raw and within is not None:
        out["between_action_variance_net"] = float(np.mean(raw) - within * np.mean(inv_n))
        out["between_action_note"] = "raw minus the expected within-action share over the same roots and candidates"
    else:
        out["between_action_variance_net"] = None
        out["between_action_note"] = "raw between-candidate variance includes continuation noise that one repeat cannot separate"
    out["paired"] = paired_gaps(d, pop)
    return out


def faults(d: dict) -> dict:
    """Faults per phase over the branch records (repeats with a record, failed or not)."""
    rec = d["cand_repeat_state"] > D.REPEAT_NONE
    out = {"branch_records": int(rec.sum()), "failed_repeats": int((d["cand_repeat_state"] == D.REPEAT_FAILED).sum()),
           "note": "continuation = cand_faults (live play after the root turn only); prefix = replaying the recorded "
                   "prefix; root = the discarded root-turn replies the bots computed",
           "failed_note": "failed repeats counts branch records without a final score; the current collector raises "
                          "instead of writing one, so 0 does not mean nothing failed (see the collector's summary)"}
    for name, key in (("continuation", "cand_faults"), ("prefix", "cand_faults_prefix"), ("root", "cand_faults_root")):
        v = d[key][rec]
        known = v >= 0
        out[name] = {"faults": int(v[known].sum()), "repeats_with_faults": int((v[known] > 0).sum()),
                     "repeats_unrecorded": int((v == D.FAULT_UNRECORDED).sum())}
    return out


def feature_collisions(d: dict) -> dict:
    xr, xc = D.scorer_inputs(d)
    roots_hit, pairs = 0, 0
    for r, idx in roots_of(d):
        c = idx[d["cand_support"][idx]]
        seen = {}
        hit = False
        for i in c:
            key = xc[i].tobytes()
            if key in seen and d["cand_key"][seen[key]] != d["cand_key"][i]:
                pairs += 1
                hit = True
            seen.setdefault(key, i)
        roots_hit += hit
    return {"colliding_pairs": pairs, "roots_with_collisions": roots_hit}


def coverage(d: dict) -> dict:
    flags = d["cand_flags"].astype(np.int64)
    inc = d["root_incumbent"]
    labeled = d["cand_n"] > 0
    change = []
    for r, idx in roots_of(d):
        alt = idx[(idx != inc[r]) & d["cand_support"][idx] & labeled[idx]]
        change.append(bool(labeled[inc[r]] and d["cand_support"][inc[r]] and len(alt)))
    k = d["root_count"].astype(np.int64)
    return {"roots": int(len(inc)), "maps": int(len(np.unique(d["root_map"]))),
            "candidates": int(len(flags)), "k_per_root": {str(a): int(b) for a, b in sorted(Counter(k.tolist()).items())},
            "supported": int(d["cand_support"].sum()), "unsupported": int((~d["cand_support"]).sum()),
            "without_outcome": int((~labeled).sum()),
            "failed_repeats": int((d["cand_repeat_state"] == D.REPEAT_FAILED).sum()),
            "incumbent_without_outcome": int((~labeled[inc]).sum()),
            "incumbent_unsupported": int((~d["cand_support"][inc]).sum()), "changeable_roots": int(sum(change)),
            "flags": {name: int(((flags & bit) > 0).sum()) for bit, name in F.FLAG_NAMES.items()},
            "roots_with_carrier_scores": int(np.isfinite(d["cand_carrier_score"][inc]).sum()),
            "shortlist_k": {str(a): int(b) for a, b in sorted(Counter(d["root_shortlist_k"].astype(np.int64).tolist()).items())},
            "sources": dict(Counter(d["cand_source"].tolist()))}


def replay(d: dict) -> dict:
    inc = d["root_incumbent"]
    ok = (d["cand_n"][inc] > 0) & np.isfinite(d["root_original_outcome"])
    if not ok.any():
        return {"status": "unavailable (no recorded original outcome)"}
    diff = d["cand_y"][inc][ok] - d["root_original_outcome"][ok]
    step = 0.5 / np.maximum(d["root_teams"][ok] - 1, 1)
    return {"status": "memory/replay diagnostic, not repeat noise", "roots": int(ok.sum()), "mean": float(diff.mean()),
            "sd": float(diff.std(ddof=1)) if ok.sum() > 1 else None,
            "share_beyond_one_step": float((np.abs(diff) > step).mean())}


def power(d: dict, maps_plan=(300, 400, 500), shares=(0.1, 0.25, 0.5)) -> dict:
    """Map-level variance of the always-switch rule (first supported labeled alternative in key order)."""
    inc = d["root_incumbent"]
    vals, maps, teams = [], [], []
    for r, idx in roots_of(d):
        alt = idx[(idx != inc[r]) & d["cand_support"][idx] & (d["cand_n"][idx] > 0)]
        if d["cand_n"][inc[r]] == 0 or not len(alt):
            continue
        a = alt[np.argsort(d["cand_key"][alt], kind="stable")[0]]
        vals.append(d["cand_y"][a] - d["cand_y"][inc[r]])
        maps.append(d["root_map"][r])
        teams.append(d["root_teams"][r])
    if len(set(maps)) < 2:
        return {"status": "unavailable: fewer than two maps with a switchable root"}
    vals, maps, teams = np.array(vals), np.array(maps), np.array(teams)
    w = count_weights(teams)
    keys, inv = np.unique(maps, return_inverse=True)
    mw = np.bincount(inv, weights=w)
    mv = np.bincount(inv, weights=w * vals) / mw
    s2 = float(np.var(mv, ddof=1))
    return {"status": "planning approximation (map-level variance of an always-switch rule; not a test result)",
            "maps": int(len(keys)), "s2_map": s2,
            "mde": {f"maps={m} switch_share={f}": float(2.8 * np.sqrt(f * s2 / m)) for m in maps_plan for f in shares}}


def diagnose(d: dict) -> dict:
    out = {"coverage": coverage(d), "noise": repeat_stats(d), "faults": faults(d), "collisions": feature_collisions(d),
           "replay": replay(d), "power": power(d), "per_count": {}}
    for t in np.unique(d["root_teams"]).tolist():
        sub = subset_roots(d, d["root_teams"] == t)
        out["per_count"][str(t)] = {"coverage": coverage(sub), "noise": repeat_stats(sub), "faults": faults(sub)}
    return out


def subset_roots(d: dict, keep: np.ndarray) -> dict:
    """The roots `keep` and their candidates, re-indexed."""
    roots = np.nonzero(keep)[0]
    cands = np.concatenate([np.arange(d["root_first"][r], d["root_first"][r] + d["root_count"][r]) for r in roots]) \
        if len(roots) else np.zeros(0, np.int64)
    out = {k: v[roots] for k, v in d.items() if k.startswith("root_")}
    out |= {k: v[cands] for k, v in d.items() if k.startswith("cand_")}
    remap = np.full(len(d["cand_root"]), -1, np.int64)
    remap[cands] = np.arange(len(cands))
    rroot = np.full(len(d["root_first"]), -1, np.int64)
    rroot[roots] = np.arange(len(roots))
    out["root_first"] = remap[d["root_first"][roots]].astype(np.int32)
    out["root_incumbent"] = remap[d["root_incumbent"][roots]].astype(np.int32)
    out["cand_root"] = rroot[d["cand_root"][cands]].astype(np.int32)
    out["cand_dup_of"] = np.where(out["cand_dup_of"] >= 0, remap[np.maximum(out["cand_dup_of"], 0)], -1).astype(np.int32)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", nargs="*", default=["train", "val"])
    ap.add_argument("--selection", help="frozen selection (required to read the test split)")
    ap.add_argument("--out", required=True, help="output path without extension")
    a = ap.parse_args(argv)
    if "test" in a.splits:
        if not a.selection:
            raise SystemExit("the test split is read only after a frozen selection (--selection)")
        from training.evaluate_decisions import verify_selection  # noqa: PLC0415
        verify_selection(json.loads(Path(a.selection).read_text(encoding="utf-8")), Path(a.data))
    index = D.index_of(Path(a.data))
    report = {"authoritative": index["authoritative"], "exploratory_reasons": index["exploratory_reasons"],
              "drops": index["drops"]["by_reason"], "splits": {}}
    for s in a.splits:
        report["splits"][s] = diagnose(D.load(Path(a.data), s, test_unlocked=s == "test"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    lines = ["# Decision data: coverage, noise and power", ""]
    if not report["authoritative"]:
        lines += ["**Exploratory data:** " + "; ".join(report["exploratory_reasons"]) + ".", ""]
    lines += ["No rule here kills or passes anything; these numbers size the next collection.", "",
              "| Split | Roots | Maps | Changeable | Without outcome | Failed repeats | Noise roots | Within-action var | "
              "Between-action var (raw / net) | Paired gap var (implied cov) | Faults prefix / root / continuation | Collisions |",
              "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|---:|"]
    for s, r in report["splits"].items():
        c, n, fl = r["coverage"], r["noise"], r["faults"]
        f = lambda v: "-" if v is None else f"{v:.5f}"  # noqa: E731
        pg = n["paired"]
        lines.append(f"| {s} | {c['roots']} | {c['maps']} | {c['changeable_roots']} | {c['without_outcome']} | "
                     f"{c['failed_repeats']} | {n['noise_roots']} | {f(n['within_action_variance'])} | "
                     f"{f(n['between_action_variance_raw'])} / {f(n['between_action_variance_net'])} | "
                     f"{f(pg['paired_gap_variance'])} ({f(pg.get('implied_covariance'))}) | "
                     f"{fl['prefix']['faults']} / {fl['root']['faults']} / {fl['continuation']['faults']} | "
                     f"{r['collisions']['colliding_pairs']} |")
    lines += ["", "Assumptions:", ""] + [f"- {x}" for x in NOISE_ASSUMPTIONS]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[decision_noise] wrote {out}.json/.md")


if __name__ == "__main__":
    main()
