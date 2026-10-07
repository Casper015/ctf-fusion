"""Offline evaluation report for one model of the matrix (Markdown + JSON).

    python -m training.evaluate_offline --model n08 --policy runs/policy_n08 --value runs/value_n08 \
        [--quantized inference/models/weights_n08.npz] --out reports/n08

The model is evaluated on every team count it serves (profiles/); the general
model on the counts without a dedicated model, unless --teams says otherwise.

Everything that is fitted here (inertia rate, mixture weight alpha) is fitted
on --fit-split (default val) and reported on --split (default test), so the
reported losses are held out. Predictors of the opponent joint move:

  uniform     uniform over legal joint moves
  inertia     each unit repeats its last displacement with a fitted rate (Player2's ply-2 rule)
  heuristic   the legacy planner forecast q_H (training.heuristic_baseline), product of unit marginals
  learned     q_theta, float, temperature-calibrated
  quantized   q_theta as the C++ computes it (integer export)
  mixture     (1 - alpha) q_H + alpha q_theta, alpha fitted per team count

Safe fallback: a team count gets `use: true` only if the mixture beats the
baseline (heuristic, else inertia) on the evaluation split by --min-gain nats.
The exporter writes alpha = 0 for every other count, and alpha = 0 is exactly
the legacy bot.

Offline log-loss says nothing about match strength; paired matches decide.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion import metrics
from fusion.labels import STRATA
from fusion.mixture import fit_alpha, mixture_loss
from fusion.models import JOINT, QuantizedTier, policy_batch, value_inputs
from fusion.nn import masked_log_softmax
from fusion.paths import model_counts, served_counts
from fusion.schema import layout as feature_layout
from training.common import batches, policy_rows
from training.dataset import load
from training.export_weights import load_policy, load_value

CONT = {(0, 0): 0, (0, -1): 1, (0, 1): 2, (-1, 0): 3, (1, 0): 4}


def cont_move(rec, layout) -> np.ndarray:
    u = layout.get(rec, "u")
    col = {n: i for i, n in enumerate(layout.names("unit"))}
    dx, dy = u[..., col["last_dx"]], u[..., col["last_dy"]]
    out = np.zeros(dx.shape, np.int64)
    for (x, y), m in CONT.items():
        out[(dx == x) & (dy == y)] = m
    return out


def fit_inertia(d, layout) -> dict[int, float]:
    legal = layout.get(d["rec"], "legal").astype(bool)
    cont = cont_move(d["rec"], layout)
    r = np.arange(len(cont))[:, None]
    eligible = legal[r, np.arange(3)[None, :], cont] & (legal.sum(-1) >= 2)
    hit = d["y_move"] == cont
    w = np.repeat(d["weight"][:, None], 3, 1)
    out = {}
    for t in np.unique(d["teams"]):
        sel = eligible & (d["teams"] == t)[:, None]
        out[int(t)] = float((w * hit)[sel].sum() / max(w[sel].sum(), 1e-12))
    return out


def unit_inertia(rec, layout, teams, rho: dict[int, float]) -> np.ndarray:
    legal = layout.get(rec, "legal").astype(np.float64)
    cont = cont_move(rec, layout)
    n_legal = legal.sum(-1, keepdims=True)
    q = legal / n_legal
    p = np.array([rho.get(int(t), 0.0) for t in teams])[:, None]
    r = np.arange(len(rec))[:, None]
    k = np.arange(3)[None, :]
    ok = (legal[r, k, cont] > 0) & (n_legal[..., 0] >= 2)
    other = legal * ((1 - p) / np.maximum(n_legal[..., 0] - 1, 1))[..., None]
    other[r, k, cont] = p
    return np.where(ok[..., None], other, q)


def unit_heuristic(d, layout) -> tuple[np.ndarray, float]:
    """q_H unit marginals; units without a forecast fall back to uniform over legal moves."""
    legal = layout.get(d["rec"], "legal").astype(np.float64)
    uniform = legal / legal.sum(-1, keepdims=True)
    qh = d["qh"].astype(np.float64)
    missing = np.isnan(qh).any(-1)
    origin = layout.get(d["rec"], "origin")
    return np.where(missing[..., None], uniform, qh), float(missing[origin >= 0].mean()) if (origin >= 0).any() else 0.0


def joint_of_units(q: np.ndarray, y_move: np.ndarray) -> np.ndarray:
    r = np.arange(len(q))[:, None]
    return np.prod(q[r, np.arange(3)[None, :], y_move], axis=1)


def learned_joint(model, d, layout, quantized=None):
    """(p at label, top-1 confidence, top-1 correct, unit marginals) per row."""
    p_true, conf, correct, marg = [], [], [], []
    for idx in batches(len(d["rec"]), 4096, None):
        rec = d["rec"][idx]
        if quantized is not None:
            logits, mask = quantized.joint_logits(rec, layout)
        else:
            X = policy_batch(rec, layout)
            logits, mask = model.joint_logits(X), X.joint_mask
        logp = masked_log_softmax(logits.astype(np.float64), mask)
        p = np.where(mask, np.exp(logp), 0.0)
        y = d["y_joint"][idx].astype(np.int64)
        p_true.append(p[np.arange(len(y)), y])
        conf.append(p.max(1))
        correct.append(p.argmax(1) == y)
        P = p.reshape(-1, 5, 5, 5)
        marg.append(np.stack([P.sum((2, 3)), P.sum((1, 3)), P.sum((1, 2))], 1))
    return np.concatenate(p_true), np.concatenate(conf), np.concatenate(correct), np.concatenate(marg)


def heads(model, d, layout):
    """Act-head and target-head log-loss / accuracy at the requested move."""
    la, lt = [], []
    for idx in batches(len(d["rec"]), 4096, None):
        X = policy_batch(d["rec"][idx], layout)
        move = d["y_move"][idx].astype(np.int64)
        lpa = masked_log_softmax(model.act_logits(X, move).astype(np.float64), X.act_mask(move))
        ya = d["y_act"][idx].astype(np.int64)
        ok = ya >= 0
        w = np.repeat(d["weight"][idx][:, None], 3, 1)
        la.append((np.take_along_axis(lpa, np.maximum(ya, 0)[..., None], -1)[..., 0][ok], lpa.argmax(-1)[ok] == ya[ok], w[ok]))
        lpt = masked_log_softmax(model.target_logits(X, move).astype(np.float64), X.cand_valid)
        yt = d["y_tgt"][idx].astype(np.int64)
        ok = yt >= 0
        lt.append((np.take_along_axis(lpt, np.maximum(yt, 0)[..., None], -1)[..., 0][ok], lpt.argmax(-1)[ok] == yt[ok], w[ok]))
    out = {}
    for name, parts in (("act", la), ("target", lt)):
        lp, hit, w = (np.concatenate([p[i] for p in parts]) for i in range(3))
        out[name] = {"rows": int(len(lp)), "log_loss": metrics.log_loss(np.exp(lp), w) if len(lp) else None,
                     "accuracy": metrics.rate(hit, w) if len(lp) else None}
    return out


def policy_section(model, d_fit, d_eval, layout, quantized, min_gain=0.01):
    rho = fit_inertia(d_fit, layout)
    have_qh = "qh" in d_eval and "qh" in d_fit
    preds = {}
    w = d_eval["weight"].astype(np.float64)
    jm = np.concatenate([policy_batch(d_eval["rec"][i], layout).joint_mask for i in batches(len(d_eval["rec"]), 8192, None)])
    preds["uniform"] = 1.0 / jm.sum(1)
    preds["inertia"] = joint_of_units(unit_inertia(d_eval["rec"], layout, d_eval["teams"], rho), d_eval["y_move"])
    missing = None
    if have_qh:
        qh_units, missing = unit_heuristic(d_eval, layout)
        preds["heuristic"] = joint_of_units(qh_units, d_eval["y_move"])
    p_true, conf, correct, marg = learned_joint(model, d_eval, layout)
    preds["learned"] = p_true
    if quantized is not None:
        preds["quantized"] = learned_joint(None, d_eval, layout, quantized)[0]

    # Mixture weights: fit on the fit split, evaluate here.
    base_name = "heuristic" if have_qh else "inertia"
    if have_qh:
        fit_base = joint_of_units(unit_heuristic(d_fit, layout)[0], d_fit["y_move"])
    else:
        fit_base = joint_of_units(unit_inertia(d_fit["rec"], layout, d_fit["teams"], rho), d_fit["y_move"])
    fit_learned = learned_joint(model, d_fit, layout)[0]
    alpha, by_teams = {}, []
    for t in sorted(np.unique(d_eval["teams"])):
        sf = d_fit["teams"] == t
        se = d_eval["teams"] == t
        a = fit_alpha(fit_base[sf], fit_learned[sf], d_fit["weight"][sf])[0] if sf.any() else None
        alpha[int(t)] = {"alpha": a, "baseline": base_name, "fit_rows": int(sf.sum())}
        row = {"teams": int(t), "rows": int(se.sum())}
        for name, p in preds.items():
            row[name] = metrics.log_loss(p[se], w[se])
        if a is not None:
            row["mixture"] = mixture_loss(a, preds[base_name][se], preds["learned"][se], w[se])
        row["alpha"] = a
        gain = row[base_name] - row["mixture"] if a is not None else None
        alpha[int(t)] |= {"use": bool(gain is not None and gain > min_gain and a > 0), "gain": gain}
        row["use"] = alpha[int(t)]["use"]
        row["top1"] = metrics.rate(correct[se], w[se])
        row["ece"] = metrics.ece(conf[se], correct[se].astype(float), w[se])
        by_teams.append(row)

    origin = feature_layout().get(d_eval["rec"], "origin")
    active = origin >= 0
    unit_hit = (marg.argmax(-1) == d_eval["y_move"]) & active
    by_stratum = []
    for s, name in enumerate(STRATA):
        sel = d_eval["stratum"] == s
        if sel.any():
            by_stratum.append({"stratum": name, "rows": int(sel.sum())} |
                              {k: metrics.log_loss(p[sel], w[sel]) for k, p in preds.items()})
    return {
        "rows": int(len(w)), "inertia_rate": rho, "alpha": alpha, "by_teams": by_teams, "by_stratum": by_stratum,
        "unit_move_accuracy": float((unit_hit.sum(1) * w).sum() / max((active.sum(1) * w).sum(), 1e-12)),
        "blocked_share": float(d_eval["blocked"][active].mean()) if active.any() else 0.0,
        "heuristic_missing_units": missing, "heads": heads(model, d_eval, layout),
    }


def value_section(model, beta, d, layout, quantized):
    z = np.concatenate([model.logits(value_inputs(d["rec"][i], layout))[:, 0] for i in batches(len(d["rec"]), 8192, None)])
    p = 1 / (1 + np.exp(-beta * z.astype(np.float64)))
    y = d["v_y"][:, 0].astype(np.float64)
    w = d["weight"].astype(np.float64)
    group = d["match"].astype(np.int64) * 1024 + d["turn"].astype(np.int64)
    rank, pairs = metrics.pair_ranking_accuracy(p, y, group)
    out = {"rows": int(len(y)), "brier": metrics.brier(p, y, w), "brier_constant": metrics.brier(np.full_like(y, 0.5), y, w),
           "bce": metrics.bce(p, y, w), "ranking_accuracy": rank, "ranking_pairs": pairs,
           "calibration_error": metrics.binned_calibration(p, y, w)}
    if quantized is not None:
        pq = np.concatenate([quantized.value_prob(d["rec"][i], layout) for i in batches(len(d["rec"]), 8192, None)])
        out["brier_quantized"] = metrics.brier(pq, y, w)
        out["max_abs_quantization_error"] = float(np.abs(pq - p).max())
    phases = []
    for name, lo, hi in (("turns 1-133", 1, 133), ("turns 134-266", 134, 266), ("turns 267-400", 267, 400)):
        sel = (d["turn"] >= lo) & (d["turn"] <= hi)
        if sel.any():
            phases.append({"phase": name, "rows": int(sel.sum()), "brier": metrics.brier(p[sel], y[sel], w[sel]),
                           "ranking_accuracy": metrics.pair_ranking_accuracy(p[sel], y[sel], group[sel])[0]})
    out["by_phase"] = phases
    return out


def markdown(model, split, pol, val, sizes) -> str:
    def f(v):
        return "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)

    names = [k for k in ("uniform", "inertia", "heuristic", "learned", "quantized", "mixture") if any(k in r for r in pol["by_teams"])]
    lines = [f"# Offline evaluation: model {model}, split {split}", "",
             "Log-loss is nats per joint opponent move (lower is better), weighted back to the natural turn "
             "distribution. alpha is fitted on the fit split and applied here. Offline numbers rank predictors; "
             "only paired matches measure strength.", "", "## Opponent joint-move prediction", "",
             "| teams | rows | " + " | ".join(names) + " | alpha | use | top-1 | ECE |",
             "|---:|---:|" + "---:|" * (len(names) + 4)]
    for r in pol["by_teams"]:
        lines.append(f"| {r['teams']} | {r['rows']} | " + " | ".join(f(r.get(n)) for n in names) +
                     f" | {f(r['alpha'])} | {'yes' if r['use'] else 'no'} | {f(r['top1'])} | {f(r['ece'])} |")
    lines += ["", f"Mixture baseline: {pol['alpha'][next(iter(pol['alpha']))]['baseline'] if pol['alpha'] else '-'}. "
              f"Inertia rates: {', '.join(f'{t}: {v:.3f}' for t, v in pol['inertia_rate'].items())}. "
              f"Unit move accuracy (marginal argmax): {pol['unit_move_accuracy']:.4f}. "
              f"Blocked requested moves: {pol['blocked_share']:.4f}."
              + (f" Units without a heuristic forecast: {pol['heuristic_missing_units']:.4f}." if pol["heuristic_missing_units"] is not None else ""),
              "", "### By stratum", "", "| stratum | rows | " + " | ".join(n for n in names if n != "mixture") + " |",
              "|---|---:|" + "---:|" * len([n for n in names if n != "mixture"])]
    for r in pol["by_stratum"]:
        lines.append(f"| {r['stratum']} | {r['rows']} | " + " | ".join(f(r.get(n)) for n in names if n != "mixture") + " |")
    h = pol["heads"]
    lines += ["", "### Action heads (at the requested move)", "", "| head | rows | log-loss | accuracy |", "|---|---:|---:|---:|",
              f"| act | {h['act']['rows']} | {f(h['act']['log_loss'])} | {f(h['act']['accuracy'])} |",
              f"| target | {h['target']['rows']} | {f(h['target']['log_loss'])} | {f(h['target']['accuracy'])} |",
              "", "## Board value", "",
              f"Rows {val['rows']}. Brier {f(val['brier'])} (constant 0.5: {f(val['brier_constant'])}), "
              f"BCE {f(val['bce'])}, binned calibration error {f(val['calibration_error'])}, pairwise ranking accuracy {f(val['ranking_accuracy'])} over {val['ranking_pairs']} pairs."
              + (f" Quantized Brier {f(val['brier_quantized'])}, max |p_q - p| {val['max_abs_quantization_error']:.2e}."
                 if "brier_quantized" in val else ""),
              "", "| phase | rows | Brier | ranking accuracy |", "|---|---:|---:|---:|"]
    for r in val["by_phase"]:
        lines.append(f"| {r['phase']} | {r['rows']} | {f(r['brier'])} | {f(r['ranking_accuracy'])} |")
    if sizes:
        lines += ["", "## Model size", "", "| net | parameters | MACs per call |", "|---|---:|---:|"]
        for k in sizes["params"]:
            lines.append(f"| {k} | {sizes['params'][k]} | {sizes['macs'][k]} |")
        lines.append(f"\nJoint policy for one opponent team: {sizes['macs']['joint_policy_per_team']} MACs "
                     "(15 mover calls + 1 squad call).")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--model", required=True)
    ap.add_argument("--teams", type=int, nargs="*", help="team counts to evaluate (default: those the model serves)")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--quantized", help="exported weights_<model>.npz to evaluate the integer model too")
    ap.add_argument("--min-gain", type=float, default=0.01, help="nats the mixture must beat the baseline by to be used")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--out", required=True, help="output path without extension (.md and .json are written)")
    a = ap.parse_args(argv)
    layout = feature_layout()
    policy = load_policy(Path(a.policy), layout)
    policy.calibrated = True
    value, beta = load_value(Path(a.value), layout)
    quantized = QuantizedTier(dict(np.load(a.quantized))) if a.quantized else None
    counts = a.teams or served_counts(a.model) or model_counts(a.model)
    d_eval_all = load(a.shards, counts, a.split)
    d_fit_all = load(a.shards, counts, a.fit_split)
    pick = lambda d: {k: v[policy_rows(d, a.shards)] for k, v in d.items()}  # real opponents only
    pol = policy_section(policy, pick(d_fit_all), pick(d_eval_all), layout, quantized, a.min_gain)
    val = value_section(value, beta, d_eval_all, layout, quantized)
    sizes = None
    manifest = Path(a.quantized).parent / "MANIFEST.json" if a.quantized else None
    if manifest and manifest.exists():
        sizes = json.loads(manifest.read_text(encoding="utf-8")).get(a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown(a.model, a.split, pol, val, sizes), encoding="utf-8")
    report = {"model": a.model, "team_counts": counts, "split": a.split, "fit_split": a.fit_split,
              "alpha": {str(k): v for k, v in pol["alpha"].items()}, "policy": pol, "value": val}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
