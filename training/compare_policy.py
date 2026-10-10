"""Offline comparison of opponent-policy runs on the same rows, stratified by situation (P feature ablation).

    python -m training.compare_policy --shards data/shards_l3 --split test --model general \\
        --run p2=runs/policy_xxl_p2_s1 --run p3=runs/policy_xxl_p3_s1 --compare p3:p2 \\
        --out reports/models/p_features/test_s1

Every run is scored on the identical opponent-policy rows (valid replies of labelled families; our own
designated carrier never counts) of one split of layout-3 shards, with its calibrated temperatures. Per row:
the joint move log loss (-log q(label)), the act head's and the target head's mean log loss over the units
that have such a label, and top-1 joint accuracy.

Strata (computed from the record, the same for every run):
  contact       an own unit has an enemy within Manhattan 3
  pursuit       an enemy carrier is within 6 of an own unit (p3 chase_present at stay)
  retreat       an own unit could be killed by shooters able to reach it (p3 lethal_reach at stay)
  own carrier   an own unit carries a flag
  carrier shot  an enemy carrier can be shot from some legal destination (p3 focus_present)
  distance      nearest enemy (Manhattan) over own units: 0-2, 3-4, 5-8, 9+
  teams         team count

--compare X:Y gives X - Y per stratum with 95% intervals clustered by map. Better offline prediction is a
prerequisite, not evidence of stronger play: compare at equal full-turn time in games next.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion.nn import masked_log_softmax
from fusion.paths import model_counts
from fusion.schema import layout as feature_layout
from training.common import batches, policy_rows
from training.dataset import load, load_index
from training.evaluate_value import clustered_mean_ci
from training.export_weights import load_policy

LABELS = ("y_move", "y_joint", "y_act", "y_tgt")


def row_metrics(model, d: dict, layout, size: int = 4096) -> dict[str, np.ndarray]:
    joint, acc, act, tgt = [], [], [], []
    for idx in batches(len(d["rec"]), size, None):
        X = model.batch(d["rec"][idx], layout)
        lp = masked_log_softmax(model.joint_logits(X).astype(np.float64), X.joint_mask)
        y = d["y_joint"][idx].astype(np.int64)
        joint.append(-lp[np.arange(len(y)), y])
        acc.append((lp.argmax(1) == y).astype(np.float64))
        move = d["y_move"][idx].astype(np.int64)
        la = masked_log_softmax(model.act_logits(X, move).astype(np.float64), X.act_mask(move))
        ya = d["y_act"][idx].astype(np.int64)
        lt = masked_log_softmax(model.target_logits(X, move).astype(np.float64), X.cand_valid)
        yt = d["y_tgt"][idx].astype(np.int64)
        for logp, lab, out in ((la, ya, act), (lt, yt, tgt)):
            ok = lab >= 0
            nll = -np.take_along_axis(logp, np.maximum(lab, 0)[..., None], -1)[..., 0]
            n = ok.sum(1)
            out.append(np.where(n > 0, np.where(ok, nll, 0).sum(1) / np.maximum(n, 1), np.nan))
    return {"joint_nll": np.concatenate(joint), "joint_top1": np.concatenate(acc), "act_nll": np.concatenate(act),
            "target_nll": np.concatenate(tgt)}


def strata(rec: np.ndarray, teams: np.ndarray, layout) -> dict[str, np.ndarray]:
    u = layout.get(rec, "u")
    ui = {n: i for i, n in enumerate(layout.names("unit"))}
    pc = layout.get(rec, "pc")
    ci = {n: i for i, n in enumerate(layout.names("combat"))}
    active = u[..., ui["active"]] > 0
    near = np.where(active, u[..., ui["nearest_enemy"]], 99).min(1)
    out = {"all": np.ones(len(rec), bool),
           "contact": (active & (u[..., ui["enemies_r3"]] > 0)).any(1),
           "pursuit": (pc[:, :, 0, ci["chase_present"]] > 0).any(1),
           "retreat": (pc[:, :, 0, ci["lethal_reach"]] > 0).any(1),
           "own carrier": (u[..., ui["carrying"]] > 0).any(1),
           "carrier shot": (pc[..., ci["focus_present"]] > 0).any((1, 2)),
           "dist 0-2": near <= 2, "dist 3-4": (near >= 3) & (near <= 4), "dist 5-8": (near >= 5) & (near <= 8),
           "dist 9+": near >= 9}
    for t in np.unique(teams):
        out[f"teams {int(t)}"] = teams == t
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", required=True, help="layout-3 shards (they provide both p2 and p3)")
    ap.add_argument("--model", default="general")
    ap.add_argument("--teams", type=int, nargs="*")
    ap.add_argument("--split", default="test")
    ap.add_argument("--run", action="append", required=True, help="NAME=POLICY_RUN_DIR")
    ap.add_argument("--compare", action="append", default=[], help="X:Y paired difference X - Y")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    L = feature_layout()
    counts = a.teams or model_counts(a.model)
    d = load(a.shards, counts, a.split, families=("p2", "p3"))
    keep = policy_rows(d, a.shards)
    d = {k: v[keep] for k, v in d.items()}
    index = load_index(a.shards)
    maps = np.array([m["map_seed"] for m in index["matches"]], np.int64)[d["match"]]
    w = d["weight"].astype(np.float64)
    groups = strata(d["rec"], d["teams"], L)
    report = {"split": a.split, "rows": int(len(w)), "maps": int(len(np.unique(maps))), "runs": {}, "compare": []}
    per_row = {}
    for spec in a.run:
        name, path = spec.split("=", 1)
        model = load_policy(Path(path), L)
        model.calibrated = True
        m = row_metrics(model, d, L)
        per_row[name] = m
        cfg = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
        report["runs"][name] = {"family": model.family, "hidden": model.hidden, "seed": cfg.get("seed"), "strata": {
            g: {k: (float(np.nansum(w[mask] * v[mask]) / max(w[mask & ~np.isnan(v)].sum(), 1e-12)) if mask.any() else None)
                for k, v in m.items()} | {"rows": int(mask.sum())} for g, mask in groups.items()}}
    for spec in a.compare:
        x, y = spec.split(":")
        res = {}
        for g, mask in groups.items():
            res[g] = {}
            for k in ("joint_nll", "act_nll", "target_nll"):
                diff = per_row[x][k] - per_row[y][k]
                ok = mask & ~np.isnan(diff)
                res[g][k] = clustered_mean_ci(diff[ok], w[ok], maps[ok]) if ok.any() else None
        report["compare"].append({"x": x, "y": y, "strata": res})
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    lines = [f"# Opponent policy comparison: {a.split} split", "",
             f"{report['rows']} opponent rows from {report['maps']} maps; calibrated models. Differences X - Y with 95% "
             "intervals clustered by map (negative log loss difference: X predicts better). Better prediction is not "
             "evidence of stronger play.", ""]
    for c in report["compare"]:
        lines += [f"## {c['x']} - {c['y']}", "", "| Stratum | Rows | Joint NLL | Act NLL | Target NLL |", "|---|---:|---|---|---|"]
        for g, r in c["strata"].items():
            cell = lambda s: "-" if not s else f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] {s['verdict']}"  # noqa: E731
            lines.append(f"| {g} | {int(groups[g].sum())} | {cell(r['joint_nll'])} | {cell(r['act_nll'])} | {cell(r['target_nll'])} |")
        lines.append("")
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
