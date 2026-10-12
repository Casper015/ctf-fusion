"""Noise, coverage and power diagnostics of a decision dataset (task T0034, docs/RL_PLAN.md section 3).

    python -m training.decision_noise --data data/decision/t0010_p2 --out reports/models/decision/noise

Reported per split and per team count, never turned into an automatic kill rule:
  coverage   roots, effective (independent) maps, candidates per root (actual K), supported candidates, label
             availability (candidates without any continuation), changeable roots (incumbent labeled and at least
             one supported, labeled alternative), canonicalization flags, effective duplicates, targets outside the
             representation, feature collisions (distinct canonical orders with identical scorer features);
  noise      within-action repeat variance (pooled sample variance of repeats of the *same* candidate at the same
             root), only where two or more repeats exist: one repeat cannot establish repeat variance; between-
             action variance of candidate means inside a root, raw, and with the expected within-action share
             removed when that share is known. Spread of effects across *different* roots is never used as
             continuation noise;
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


def repeat_stats(d: dict) -> dict:
    n = d["cand_n"].astype(np.int64)
    out = {"repeats": {str(k): int(v) for k, v in sorted(Counter(n.tolist()).items())}}
    m = (n >= 2) & np.isfinite(d["cand_var"])
    if m.any():
        dof = (n[m] - 1).sum()
        out["within_action_variance"] = float(((n[m] - 1) * d["cand_var"][m]).sum() / dof)
        out["within_action_dof"] = int(dof)
    else:
        out["within_action_variance"] = None
        out["within_action_note"] = "unavailable: one repeat cannot establish repeat variance"
    raw, inv_n = [], []
    for r, idx in roots_of(d):
        c = idx[(d["cand_support"][idx] | (idx == d["root_incumbent"][r])) & (n[idx] > 0)]
        if len(c) >= 2:
            raw.append(np.var(d["cand_y"][c], ddof=1))
            inv_n.append(np.mean(1.0 / n[c]))
    out["between_action_variance_raw"] = float(np.mean(raw)) if raw else None
    out["between_roots"] = len(raw)
    if raw and out["within_action_variance"] is not None:
        est = float(np.mean(raw) - out["within_action_variance"] * np.mean(inv_n))
        out["between_action_variance_net"] = est
        out["between_action_note"] = "raw minus the expected within-action share (pooled within variance x mean 1/n)"
    else:
        out["between_action_variance_net"] = None
        out["between_action_note"] = "raw between-candidate variance includes continuation noise that one repeat cannot separate"
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
            "supported": int(d["cand_support"].sum()), "without_outcome": int((~labeled).sum()),
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
    out = {"coverage": coverage(d), "noise": repeat_stats(d), "collisions": feature_collisions(d), "replay": replay(d),
           "power": power(d), "per_count": {}}
    for t in np.unique(d["root_teams"]).tolist():
        sub = subset_roots(d, d["root_teams"] == t)
        out["per_count"][str(t)] = {"coverage": coverage(sub), "noise": repeat_stats(sub)}
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
              "| Split | Roots | Maps | Changeable | Without outcome | Within-action var | Between-action var (raw / net) | Collisions |",
              "|---|---:|---:|---:|---:|---|---|---:|"]
    for s, r in report["splits"].items():
        c, n = r["coverage"], r["noise"]
        f = lambda v: "-" if v is None else f"{v:.5f}"  # noqa: E731
        lines.append(f"| {s} | {c['roots']} | {c['maps']} | {c['changeable_roots']} | {c['without_outcome']} | "
                     f"{f(n['within_action_variance'])} | {f(n['between_action_variance_raw'])} / {f(n['between_action_variance_net'])} | "
                     f"{r['collisions']['colliding_pairs']} |")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[decision_noise] wrote {out}.json/.md")


if __name__ == "__main__":
    main()
