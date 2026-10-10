"""Audit what the feature families can and cannot tell apart, on real shards (run before changing a limit).

    python -m training.feature_audit --shards data/shards_l3 --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 \\
        --split train --out reports/models/p_features/audit

Two questions:

1. Attack-candidate coverage (the target head sees at most the nearest kCand = 8 hittable enemies within
   Manhattan 4). Over opponent-policy rows: how many unit-turns requested an attack, how many of those named a
   target inside the candidate slots, how many a hittable target outside them (labels.py keeps those as
   attacks with no slot: lost target labels), and how often all 8 slots were full. By team count.

2. Value aliases. Rows whose 54 v2 inputs are identical are one state to a v2 network. Among such groups that
   span at least two matches (identical states inside one match are mostly consecutive quiet turns), how far
   apart are their outcomes, and how many does v3 split (identical v2 inputs, different v3 extension)?
   Reported: rows in cross-match alias groups, the weighted outcome variance inside groups (what no v2 model
   can explain), and the share of that variance between v3 subgroups (what the extension can separate).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from fusion.protocol import ATTACK
from fusion.schema import layout as feature_layout
from training.common import policy_rows
from training.dataset import load


def coverage(d: dict, layout) -> dict:
    keep = d["valid"].copy()
    act, tgt = d["y_act"], d["y_tgt"]
    cand = layout.get(d["rec"], "cand")
    out = {}
    for t in [None, *np.unique(d["teams"])]:
        rows = keep if t is None else keep & (d["teams"] == t)
        a = (act[rows] == ATTACK)
        named = a & (tgt[rows] >= 0)
        lost = a & (tgt[rows] < 0)
        full = (cand[rows][..., -1] >= 0)
        out["all" if t is None else str(int(t))] = {
            "unit_turns": int(a.size), "attacks": int(a.sum()), "target_in_slots": int(named.sum()),
            "target_beyond_slots": int(lost.sum()), "lost_share": float(lost.sum() / max(a.sum(), 1)),
            "slots_full_share": float(full.mean()) if full.size else None}
    return out


def aliases(d: dict, layout) -> dict:
    v2 = np.ascontiguousarray(layout.get(d["rec"], "v"))
    vx = np.ascontiguousarray(layout.get(d["rec"], "vx")) if d["rec"].shape[-1] >= layout.width else None  # layout-3 records
    y = d["v_y"][:, 0].astype(np.float64)
    w = d["weight"].astype(np.float64)
    groups = defaultdict(list)
    for i, row in enumerate(v2):
        groups[row.tobytes()].append(i)
    rows_in, within, between, total_var_w = 0, 0.0, 0.0, 0.0
    split_groups, cross = 0, 0
    for idx in groups.values():
        if len(idx) < 2 or len(set(d["match"][idx].tolist())) < 2:
            continue
        cross += 1
        idx = np.array(idx)
        rows_in += len(idx)
        ww, yy = w[idx], y[idx]
        mean = (ww * yy).sum() / ww.sum()
        within += float((ww * (yy - mean) ** 2).sum())
        if vx is not None:
            sub = defaultdict(list)
            for i in idx:
                sub[vx[i].tobytes()].append(i)
            if len(sub) > 1:
                split_groups += 1
                for s in sub.values():
                    s = np.array(s)
                    ms = (w[s] * y[s]).sum() / w[s].sum()
                    between += float(w[s].sum() * (ms - mean) ** 2)
    total_var_w = float((w * (y - (w * y).sum() / w.sum()) ** 2).sum())
    return {"rows": int(len(y)), "cross_match_alias_groups": cross, "rows_in_alias_groups": rows_in,
            "alias_row_share": rows_in / max(len(y), 1),
            "within_group_variance_share": within / max(total_var_w, 1e-12),
            "groups_split_by_v3": split_groups if vx is not None else None,
            "v3_separable_share_of_within": (between / within) if vx is not None and within > 0 else None}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--teams", type=int, nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L = feature_layout()
    d = load(a.shards, a.teams, a.split, families=("p2", "v2"))
    keep = policy_rows(d, a.shards)
    report = {"split": a.split, "coverage": coverage({k: v[keep] for k, v in d.items()}, L), "aliases": aliases(d, L)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    c, al = report["coverage"]["all"], report["aliases"]
    lines = ["# Feature audit", "", f"Split {a.split}, team counts {a.teams}.", "",
             "## Attack-candidate coverage (8 nearest hittable enemies within 4)", "",
             "| Teams | Attacks | Target in slots | Beyond slots | Lost share | Slots full |", "|---|---:|---:|---:|---:|---:|"]
    for t, r in report["coverage"].items():
        lines.append(f"| {t} | {r['attacks']} | {r['target_in_slots']} | {r['target_beyond_slots']} | {r['lost_share']:.4f} | "
                     f"{r['slots_full_share'] if r['slots_full_share'] is None else round(r['slots_full_share'], 4)} |")
    lines += ["", "## Value aliases (identical v2 inputs across matches)", "",
              f"- rows: {al['rows']}; rows in cross-match alias groups: {al['rows_in_alias_groups']} ({al['alias_row_share']:.4f})",
              f"- outcome variance inside alias groups, as a share of all outcome variance: {al['within_group_variance_share']:.4f}",
              f"- alias groups that v3 splits: {al['groups_split_by_v3']}; share of the within-group variance between v3 "
              f"subgroups: {al['v3_separable_share_of_within']}", "", f"Total attack labels checked: {c['attacks']}."]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
