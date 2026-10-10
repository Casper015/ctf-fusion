"""Evaluate value networks for decision quality, not only scalar accuracy (the V pilot's offline report).

    python -m training.evaluate_value --cache data/vcache/v12xxl_v3 --split test \\
        --run A=runs/v_pilot_A_s1 --run B=runs/v_pilot_B_s1 --run C=runs/v_pilot_C_s1 --run D=runs/v_pilot_D_s1 \\
        --rank data/rank/v12xxl --compare B:A --compare D:C --compare C:A --compare D:B \\
        --out reports/models/v_pilot/test_s1

Every run is scored on the same rows (a cache carries both families' source records only through its own
family, so give each run a cache of its family with --cache-for v2=DIR; rows must coincide: same shards,
same split, checked by row count and match/turn columns).

Per run:
  - outcome log loss (BCE of the calibrated probability, sigmoid(beta z)), Brier score, expected calibration
    error (10 equal-width bins) and the calibration table; all weighted by the row weights;
  - the same by game phase (turns 1-100, 101-200, 201-300, 301-400) and by team count;
  - subsets (training.value_cache tags): carrier, drop, interception, close score;
  - learning curve and training effort from the run directory (log.jsonl, config.json): epochs, best epoch,
    seconds, rows, parameters, peak device memory when recorded;
  - with --rank: candidate-pair accuracy on resolved pairs, and the estimated decision regret per root,
    mean(max_c y_c - y_(argmax_c V)), next to the on-policy candidate's regret and a uniform choice's
    expected regret. y_c is the mean continuation outcome of candidate c (one or a few continuations, so
    regret is itself an estimate with sampling noise).

--compare X:Y reports X - Y for log loss, Brier and regret with 95% intervals clustered by map (the unit
that never straddles splits), so correlated turns of one game never count as independent samples. An
interval that includes zero is reported as "not shown". Lower prediction loss alone does not establish
stronger decisions: only games at equal full-turn time do.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion.schema import layout as feature_layout
from training.export_weights import load_value
from training.train_value import load_rank
from training.value_cache import TAGS, load_cache

PHASES = ((1, 100), (101, 200), (201, 300), (301, 400))


def probs(model, beta: float, v: np.ndarray, layout, size=16384) -> np.ndarray:
    out = []
    scale = layout.value_scale(model.family)
    for s in range(0, len(v), size):
        x = (np.asarray(v[s:s + size]) / scale).astype(np.float32)
        z = model.logits(x)[:, 0].astype(np.float64)
        out.append(1 / (1 + np.exp(-beta * z)))
    return np.concatenate(out) if out else np.zeros(0)


def scores(p: np.ndarray, y: np.ndarray, w: np.ndarray, bins: int = 10) -> dict:
    eps = 1e-12
    pc = np.clip(p, eps, 1 - eps)
    ll = -(y * np.log(pc) + (1 - y) * np.log(1 - pc))
    W = max(float(w.sum()), eps)
    table, ece = [], 0.0
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    for b in range(bins):
        m = idx == b
        wb = float(w[m].sum())
        if wb <= 0:
            continue
        mp, my = float((w[m] * p[m]).sum() / wb), float((w[m] * y[m]).sum() / wb)
        ece += wb / W * abs(mp - my)
        table.append({"bin": [round(edges[b], 2), round(edges[b + 1], 2)], "weight": round(wb / W, 4), "mean_p": round(mp, 4),
                      "mean_y": round(my, 4)})
    return {"rows": int(len(p)), "logloss": float((w * ll).sum() / W), "brier": float((w * (p - y) ** 2).sum() / W),
            "ece": ece, "mean_p": float((w * p).sum() / W), "mean_y": float((w * y).sum() / W), "calibration": table}


def clustered_mean_ci(values: np.ndarray, w: np.ndarray, cluster: np.ndarray, z: float = 1.96) -> dict:
    """Weighted mean of per-row values with a cluster-robust 95% interval (clusters = maps)."""
    W = float(w.sum())
    if W <= 0:
        return {"mean": None, "lo": None, "hi": None, "clusters": 0}
    mean = float((w * values).sum() / W)
    keys, inv = np.unique(cluster, return_inverse=True)
    resid = np.bincount(inv, weights=w * (values - mean)) / W
    k = len(keys)
    se = float(np.sqrt(k / max(k - 1, 1) * (resid ** 2).sum())) if k > 1 else float("nan")
    lo, hi = mean - z * se, mean + z * se
    verdict = "not shown" if not (lo > 0 or hi < 0) else ("higher" if lo > 0 else "lower")
    return {"mean": mean, "lo": lo, "hi": hi, "clusters": int(k), "verdict": verdict}


def row_losses(p, y):
    pc = np.clip(p, 1e-12, 1 - 1e-12)
    return {"logloss": -(y * np.log(pc) + (1 - y) * np.log(1 - pc)), "brier": (p - y) ** 2}


def rank_metrics(model, beta, rank: dict, layout, size=2048) -> dict:
    """Pair accuracy and decision regret over roots, from a split of training.branch_dataset."""
    z = dict(np.load(rank["path"]))
    from fusion.models import value_columns  # noqa: PLC0415
    v = value_columns(z["cand_rec"], layout, model.family)
    p = probs(model, beta, v, layout, size)
    keep = z["pair_y"] >= 0
    a, b, yy, ww = z["pair_a"][keep], z["pair_b"][keep], z["pair_y"][keep], z["pair_w"][keep]
    correct = np.where(p[a] == p[b], 0.5, ((p[a] > p[b]) == (yy == 1)).astype(np.float64))
    roots = z["cand_root"]
    regret, regret_on, regret_uniform, maps = [], [], [], []
    for r in np.unique(roots):
        c = np.nonzero(roots == r)[0]
        y = z["cand_y"][c].astype(np.float64)
        best = y.max()
        regret.append(best - y[np.argmax(p[c])])
        on = c[z["cand_on_policy"][c]]
        regret_on.append(best - float(z["cand_y"][on[0]]) if len(on) else np.nan)
        regret_uniform.append(best - y.mean())
        maps.append(int(z["root_map"][r]))
    regret, maps = np.array(regret), np.array(maps)
    regret_on = np.asarray(regret_on)
    valid_on = np.isfinite(regret_on)
    pair_maps = z["root_map"][roots[a]]
    majority = float((ww * yy).sum() / ww.sum()) if ww.sum() > 0 else None
    return {"roots": int(len(regret)), "resolved_pairs": int(keep.sum()), "uncertain_pairs": int((~keep).sum()),
            "pair_accuracy": clustered_mean_ci(correct, ww, pair_maps) if keep.any() else None,
            "pair_random_baseline": 0.5, "pair_majority_baseline": max(majority, 1 - majority) if majority is not None else None,
            "pair_majority_note": "descriptive label-frequency baseline on this split; not a fitted policy",
            "regret": clustered_mean_ci(regret, np.ones_like(regret), maps),
            "regret_vs_incumbent": clustered_mean_ci(regret[valid_on] - regret_on[valid_on],
                                                      np.ones(valid_on.sum()), maps[valid_on]),
            "incumbent_roots": int(valid_on.sum()),
            "regret_on_policy": float(regret_on[valid_on].mean()) if valid_on.any() else None,
            "regret_uniform": float(np.mean(regret_uniform)) if len(regret_uniform) else None,
            "continuation_repetitions": {"min": int(z["cand_n"].min()), "max": int(z["cand_n"].max())},
            "label_note": "sampled continuation means, not exact action values",
            "_regret_rows": regret, "_regret_maps": maps}


def run_effort(run: Path) -> dict:
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    log = [json.loads(l) for l in (run / "log.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    epochs = [r for r in log if "epoch" in r]
    best = min(epochs, key=lambda r: r["val"]) if epochs else None
    backend = cfg.get("backend", {})
    return {"hidden": cfg.get("hidden"), "family": (cfg.get("features") or {}).get("family", "v2"), "params": cfg.get("params"),
            "seed": cfg.get("seed"), "rows": cfg.get("rows"), "epochs": len(epochs), "best_epoch": best and best["epoch"],
            "best_val": best and best["val"], "seconds": cfg.get("train_seconds", sum(r.get("seconds", 0) for r in epochs)),
            "optimizer_steps": epochs[-1].get("optimizer_steps") if epochs else None,
            "fit_seconds": cfg.get("fit_seconds"), "pipeline_seconds": cfg.get("pipeline_seconds"),
            "calibration_seconds": cfg.get("calibration_seconds"), "backend_timing": backend.get("timing"),
            "curve": [{k: r[k] for k in ("epoch", "lr", "train", "val", "seconds") if k in r} for r in epochs],
            "device": backend.get("device", "numpy"), "gpu": backend.get("gpu"),
            "peak_allocated_gb": backend.get("peak_allocated_gb"), "peak_reserved_gb": backend.get("peak_reserved_gb"),
            "rank": cfg.get("rank"), "rank_weight": (cfg.get("args") or {}).get("rank_weight", 0.0)}


def markdown(report: dict) -> str:
    f = lambda v, d=4: "-" if v is None else f"{v:.{d}f}"  # noqa: E731
    out = [f"# Value evaluation: {report['split']} split", "",
           f"Rows {report['rows']} from {report['maps']} maps. Probabilities are calibrated (sigmoid(beta z)). "
           "Intervals are 95%, clustered by map; an interval that includes zero is \"not shown\". Lower loss alone does "
           "not establish stronger decisions.", "",
           "| Run | Family | Hidden | Params | Log loss | Brier | ECE | Pair acc. | Regret | Epochs (best) | Seconds | Peak GB |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|"]
    for name, r in report["runs"].items():
        e, rk = r["effort"], r.get("rank") or {}
        acc = (rk.get("pair_accuracy") or {}).get("mean")
        reg = (rk.get("regret") or {}).get("mean")
        out.append(f"| {name} | {e['family']} | {e['hidden']} | {e['params']} | {f(r['all']['logloss'])} | {f(r['all']['brier'])} | "
                   f"{f(r['all']['ece'])} | {f(acc, 3)} | {f(reg)} | {e['epochs']} ({e['best_epoch']}) | {e['seconds']} | "
                   f"{e['peak_reserved_gb'] if e['peak_reserved_gb'] is not None else '-'} |")
    if report["compare"]:
        out += ["", "## Paired differences (X - Y)", "", "| X - Y | Metric | Mean | 95% interval | Verdict |", "|---|---|---:|---|---|"]
        for c in report["compare"]:
            for metric, s in c["metrics"].items():
                if s["mean"] is not None:
                    out.append(f"| {c['x']} - {c['y']} | {metric} | {s['mean']:+.5f} | [{s['lo']:+.5f}, {s['hi']:+.5f}] | {s['verdict']} |")
    for key, title in (("phase", "By phase (turns)"), ("teams", "By team count"), ("subsets", "Subsets")):
        out += ["", f"## {title}", "", "| Run | Group | Rows | Log loss | Brier | ECE |", "|---|---|---:|---:|---:|---:|"]
        for name, r in report["runs"].items():
            for g, s in r[key].items():
                out.append(f"| {name} | {g} | {s['rows']} | {f(s['logloss'])} | {f(s['brier'])} | {f(s['ece'])} |")
    out += ["", "## Decision regret", "", "| Run | Roots | Resolved / uncertain pairs | Regret (V choice) | On-policy | Uniform |",
            "|---|---:|---|---:|---:|---:|"]
    for name, r in report["runs"].items():
        rk = r.get("rank")
        if rk:
            out.append(f"| {name} | {rk['roots']} | {rk['resolved_pairs']} / {rk['uncertain_pairs']} | {f(rk['regret']['mean'])} "
                       f"[{f(rk['regret']['lo'])}, {f(rk['regret']['hi'])}] | {f(rk['regret_on_policy'])} | {f(rk['regret_uniform'])} |")
    out += ["", "## Learning curves", ""]
    for name, r in report["runs"].items():
        rk = r.get("rank")
        if rk:
            s = rk["regret_vs_incumbent"]
            out.append(f"- **{name} versus incumbent**, paired per root: {f(s['mean'])} "
                       f"[{f(s['lo'])}, {f(s['hi'])}], {s['clusters']} map groups, {s.get('verdict', 'not shown')}. "
                       f"Pair baselines: random {rk['pair_random_baseline']}, majority {f(rk['pair_majority_baseline'])}. "
                       f"Continuations per candidate: {rk['continuation_repetitions']} (noisy estimates).")
    for name, r in report["runs"].items():
        curve = ", ".join(f"{c['epoch']}: {c['val']:.4f}" for c in r["effort"]["curve"])
        out.append(f"- **{name}** (val outcome loss by epoch): {curve}")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="V-only cache (training.value_cache) of the runs' family")
    ap.add_argument("--cache-for", action="append", default=[], help="FAMILY=DIR: cache for runs of another family")
    ap.add_argument("--split", default="test")
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (value runs)")
    ap.add_argument("--rank", help="ranking data from training.branch_dataset (the same split is used)")
    ap.add_argument("--compare", action="append", default=[], help="X:Y paired difference X - Y")
    ap.add_argument("--out", required=True, help="output path without extension (.md and .json)")
    ap.add_argument("--batch", type=int, default=2048, help="bounded FP32 inference batch, shared by all capacities")
    a = ap.parse_args(argv)
    if a.batch < 1:
        raise SystemExit("--batch must be positive")
    L = feature_layout()
    caches = {"": a.cache} | dict(x.split("=", 1) for x in a.cache_for)
    rows, ref = {}, None
    report = {"split": a.split, "runs": {}, "compare": []}
    for spec in a.run:
        name, path = spec.split("=", 1)
        model, beta = load_value(Path(path), L)
        cache = caches.get(model.family) or caches[""]
        d = load_cache(cache, a.split, model.family)
        key = tuple(np.asarray(d[k]) for k in ("match", "turn", "teams", "map", "v_y", "weight"))
        if ref is None:
            ref = key
            report |= {"rows": int(len(key[0])), "maps": int(len(np.unique(d["map"])))}
        elif not all(np.array_equal(x, y) for x, y in zip(ref, key)):
            raise SystemExit(f"run {name}: its cache's {a.split} rows differ from the first run's; score every run on the same rows")
        p = probs(model, beta, d["v"], L, a.batch)
        y = np.asarray(d["v_y"])[:, 0].astype(np.float64)
        w = np.asarray(d["weight"]).astype(np.float64)
        turn, teams, tags = np.asarray(d["turn"]), np.asarray(d["teams"]), np.asarray(d["tags"])
        r = {"all": scores(p, y, w), "effort": run_effort(Path(path)), "beta": beta,
             "phase": {f"{lo}-{hi}": scores(p[m], y[m], w[m]) for lo, hi in PHASES if (m := (turn >= lo) & (turn <= hi)).any()},
             "teams": {str(t): scores(p[teams == t], y[teams == t], w[teams == t]) for t in np.unique(teams)},
             "subsets": {n: scores(p[m], y[m], w[m]) for n, bit in TAGS.items() if (m := (tags & bit) > 0).any()}}
        if a.rank:
            rk = load_rank(a.rank, a.split, model.family, L)
            if rk is not None:
                r["rank"] = rank_metrics(model, beta, {"path": Path(a.rank) / a.split / "rank.npz"}, L, a.batch)
        rows[name] = {"losses": row_losses(p, y), "w": w, "maps": np.asarray(d["map"])}
        report["runs"][name] = r
    for spec in a.compare:
        x, y = spec.split(":")
        rx, ry = rows[x], rows[y]
        metrics = {m: clustered_mean_ci(rx["losses"][m] - ry["losses"][m], rx["w"], rx["maps"]) for m in ("logloss", "brier")}
        kx, ky = report["runs"][x].get("rank"), report["runs"][y].get("rank")
        if kx and ky:
            if not np.array_equal(kx["_regret_maps"], ky["_regret_maps"]):
                raise SystemExit("paired regret requires identical roots and map membership")
            metrics["regret"] = clustered_mean_ci(kx["_regret_rows"] - ky["_regret_rows"], np.ones(len(kx["_regret_rows"])),
                                                  kx["_regret_maps"])
        report["compare"].append({"x": x, "y": y, "metrics": metrics})
    for r in report["runs"].values():
        if r.get("rank"):
            r["rank"] = {k: v for k, v in r["rank"].items() if not k.startswith("_")}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
