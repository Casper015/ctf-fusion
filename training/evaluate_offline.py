"""Offline evaluation report and deployment gate for one model of the matrix (Markdown + JSON).

    python -m training.evaluate_offline --model n08 --policy runs/policy_n08 --value runs/value_n08 --out reports/n08 \
        [--adapter joint-v2]

The model is evaluated on every team count it serves (profiles/); the general
model on the counts without a dedicated model, unless --teams says otherwise.

Everything that is fitted here (inertia rate, mixture weight alpha) is fitted
on --fit-split (default val) and reported on --split (default test), so the
reported losses are held out. Predictors of the opponent joint move:

  uniform     uniform over legal joint moves
  inertia     each unit repeats its last displacement with a fitted rate (Player2's ply-2 rule)
  heuristic   the legacy planner forecast q_H (training.heuristic_baseline), product of unit marginals
  learned     q_theta in floating point (diagnostic only)
  quantized   q_theta as the C++ computes it: the integer model the exporter ships, temperatures folded in
  mixture     the deployment adapter (fusion/mixture.py, inference/adapter.hpp) applied to q_H and the
              quantized q_theta, alpha fitted per team count

Gate, fail closed. A team count gets `use: true` only if all of these hold:
  * every fit and eval shard carries q_H from one legacy planner (same source
    digest). Without it the legacy comparison is impossible: the inertia
    mixture is then reported as a diagnostic and the gate stays closed;
  * the mixture is the declared --adapter on the integer model;
  * the gain over q_H exceeds --min-gain nats and is more than two standard
    errors above zero, the error clustered by map;
  * the eval split has at least --min-maps maps for that team count.
The exporter writes alpha = 0 for every other count, and alpha = 0 is exactly
the legacy forecast. The report carries the artifact identity (model id,
checkpoint hashes, integer weight digest, feature schema, adapter) plus the
dataset and baseline it was measured on; training.export_weights refuses the
report for any other artifact. Value metrics are diagnostics: no gate
approves V.

Offline log-loss says nothing about match strength; paired matches decide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fusion import metrics, quant
from fusion.labels import STRATA
from fusion.mixture import ADAPTERS, adapter_inputs, adapter_prob, fit_alpha
from fusion.models import QuantizedTier, policy_batch, value_inputs
from fusion.nn import masked_log_softmax
from fusion.paths import model_counts, served_counts
from fusion.schema import layout as feature_layout
from training import export_weights
from training.common import batches, policy_rows
from training.dataset import load, load_index, load_meta

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


def at_label(q_units: np.ndarray, y_move: np.ndarray) -> np.ndarray:
    """(n, 3, 5) unit distributions -> (n, 3) probability of each unit's observed move."""
    r = np.arange(len(q_units))[:, None]
    return q_units[r, np.arange(3)[None, :], y_move.astype(np.int64)]


def map_clusters(d, index) -> np.ndarray:
    """Cluster id per row: its map seed (rows of one map are correlated), else its match."""
    maps = np.array([m.get("map_seed", f"match-{m['match_id']}") for m in index["matches"]], dtype=object)
    return np.array([str(v) for v in maps[d["match"].astype(np.int64)]])


def baseline_status(shards, counts, splits) -> tuple[bool, str, list]:
    """Is q_H present on every shard of these splits, from a single recorded legacy planner?"""
    metas = [load_meta(shards, counts, s) for s in splits]
    total, have = sum(m["shards"] for m in metas), sum(m["qh_shards"] for m in metas)
    found = sorted({b for m in metas for b in m["baselines"]})
    if have < total:
        return False, f"legacy forecast q_H missing on {total - have} of {total} shards", found
    if "unrecorded" in found:
        return False, "q_H without a recorded baseline identity (rerun training.heuristic_baseline)", found
    if len(found) != 1:
        return False, f"q_H from {len(found)} different legacy planner builds", found
    return True, "", found


def policy_section(model, tier, d_fit, d_eval, layout, adapter: str, gate: dict, qh_ok: bool, qh_reason: str,
                   clusters: np.ndarray):
    rho = fit_inertia(d_fit, layout)
    preds = {}
    w = d_eval["weight"].astype(np.float64)
    jm = np.concatenate([policy_batch(d_eval["rec"][i], layout).joint_mask for i in batches(len(d_eval["rec"]), 8192, None)])
    preds["uniform"] = 1.0 / jm.sum(1)
    inertia_eval = at_label(unit_inertia(d_eval["rec"], layout, d_eval["teams"], rho), d_eval["y_move"])
    preds["inertia"] = inertia_eval.prod(1)
    missing = None
    if "qh" in d_eval:
        qh_units, missing = unit_heuristic(d_eval, layout)
        preds["heuristic"] = at_label(qh_units, d_eval["y_move"]).prod(1)
    preds["learned"] = learned_joint(model, d_eval, layout)[0]
    q_true, conf, correct, q_marg = learned_joint(None, d_eval, layout, tier)
    preds["quantized"] = q_true
    qf_true, _, _, qf_marg = learned_joint(None, d_fit, layout, tier)

    # The mixture the gate scores is the deployed adapter on the integer model. Without a complete
    # legacy forecast the baseline falls back to inertia, as a diagnostic only.
    legacy = qh_ok and "qh" in d_fit and "qh" in d_eval
    if legacy:
        base_name = "heuristic"
        base_fit = at_label(unit_heuristic(d_fit, layout)[0], d_fit["y_move"])
        base_eval = at_label(qh_units, d_eval["y_move"])
    else:
        base_name = "inertia (diagnostic)"
        base_fit = at_label(unit_inertia(d_fit["rec"], layout, d_fit["teams"], rho), d_fit["y_move"])
        base_eval = inertia_eval
    nn_fit, nn_eval = at_label(qf_marg, d_fit["y_move"]), at_label(q_marg, d_eval["y_move"])
    alpha, by_teams = {}, []
    for t in sorted(np.unique(d_eval["teams"])):
        sf = d_fit["teams"] == t
        se = d_eval["teams"] == t
        a = None
        if sf.any():
            hq, tq = adapter_inputs(adapter, base_fit[sf], qf_true[sf], nn_fit[sf])
            a = fit_alpha(hq, tq, d_fit["weight"][sf])[0]
        row = {"teams": int(t), "rows": int(se.sum())}
        for name, p in preds.items():
            row[name] = metrics.log_loss(p[se], w[se])
        reasons = [] if legacy else [qh_reason or "no legacy forecast q_H"]
        gain = gain_se = None
        maps = len(np.unique(clusters[se]))
        if a is None:
            reasons.append("no fit rows")
        else:
            p_mix = adapter_prob(adapter, a, base_eval[se], q_true[se], nn_eval[se])
            row["mixture"] = metrics.log_loss(p_mix, w[se])
            g = np.log(np.maximum(p_mix, 1e-300)) - np.log(np.maximum(base_eval[se].prod(1), 1e-300))
            gain, gain_se, maps = metrics.clustered_mean(g, w[se], clusters[se])
            if a <= 0:
                reasons.append("alpha 0")
            if gain <= gate["min_gain"]:
                reasons.append(f"gain {gain:.4f} <= {gate['min_gain']}")
            if not gain - gate["z"] * gain_se > 0:
                reasons.append(f"gain within {gate['z']} SE")
        if maps < gate["min_maps"]:
            reasons.append(f"{maps} maps < {gate['min_maps']}")
        use = not reasons
        alpha[int(t)] = {"alpha": a, "baseline": base_name, "adapter": adapter, "fit_rows": int(sf.sum()), "use": use,
                         "gain": gain, "gain_se": gain_se, "maps": maps, "reason": "; ".join(reasons)}
        row |= {"alpha": a, "use": use, "gain": gain, "gain_se": gain_se, "maps": maps, "reason": "; ".join(reasons),
                "top1": metrics.rate(correct[se], w[se]), "ece": metrics.ece(conf[se], correct[se].astype(float), w[se])}
        by_teams.append(row)

    origin = feature_layout().get(d_eval["rec"], "origin")
    active = origin >= 0
    unit_hit = (q_marg.argmax(-1) == d_eval["y_move"]) & active
    by_stratum = []
    for s, name in enumerate(STRATA):
        sel = d_eval["stratum"] == s
        if sel.any():
            by_stratum.append({"stratum": name, "rows": int(sel.sum())} |
                              {k: metrics.log_loss(p[sel], w[sel]) for k, p in preds.items()})
    return {
        "rows": int(len(w)), "adapter": adapter, "baseline": base_name, "inertia_rate": rho, "alpha": alpha,
        "by_teams": by_teams, "by_stratum": by_stratum,
        "unit_move_accuracy": float((unit_hit.sum(1) * w).sum() / max((active.sum(1) * w).sum(), 1e-12)),
        "blocked_share": float(d_eval["blocked"][active].mean()) if active.any() else 0.0,
        "heuristic_missing_units": missing, "heads": heads(model, d_eval, layout),
    }


def value_section(model, beta, d, layout, quantized):
    """Diagnostics only: no gate approves the value network (export writes kValueApproved = false)."""
    z = np.concatenate([model.logits(value_inputs(d["rec"][i], layout))[:, 0] for i in batches(len(d["rec"]), 8192, None)])
    p = 1 / (1 + np.exp(-beta * z.astype(np.float64)))
    y = d["v_y"][:, 0].astype(np.float64)
    w = d["weight"].astype(np.float64)
    group = d["match"].astype(np.int64) * 1024 + d["turn"].astype(np.int64)
    rank, pairs = metrics.pair_ranking_accuracy(p, y, group)
    out = {"rows": int(len(y)), "brier": metrics.brier(p, y, w), "brier_constant": metrics.brier(np.full_like(y, 0.5), y, w),
           "bce": metrics.bce(p, y, w), "ranking_accuracy": rank, "ranking_pairs": pairs,
           "calibration_error": metrics.binned_calibration(p, y, w)}
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
             "distribution. alpha is fitted on the fit split and applied here. `quantized` is the integer model the "
             f"exporter ships; `mixture` is the {pol['adapter']} adapter on it. Gain = baseline minus mixture loss, "
             "± one map-clustered standard error. Offline numbers rank predictors; only paired matches measure strength.",
             "", "## Opponent joint-move prediction", "",
             "| teams | rows | maps | " + " | ".join(names) + " | alpha | gain ± SE | use | why not | top-1 | ECE |",
             "|---:|---:|---:|" + "---:|" * (len(names) + 2) + "---|---|---:|---:|"]
    for r in pol["by_teams"]:
        gain = "-" if r["gain"] is None else f"{r['gain']:+.4f} ± {r['gain_se']:.4f}"
        lines.append(f"| {r['teams']} | {r['rows']} | {r['maps']} | " + " | ".join(f(r.get(n)) for n in names) +
                     f" | {f(r['alpha'])} | {gain} | {'yes' if r['use'] else 'no'} | {r['reason'] or '-'} | "
                     f"{f(r['top1'])} | {f(r['ece'])} |")
    lines += ["", f"Mixture baseline: {pol['baseline']}. "
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
              "", "## Board value (diagnostic; no gate approves V)", "",
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
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), default="joint-v2",
                    help="deployment adapter the gate scores (inference/adapter.hpp); export must use the same")
    ap.add_argument("--min-gain", type=float, default=0.01, help="nats the mixture must beat q_H by to be used")
    ap.add_argument("--min-maps", type=int, default=20, help="eval maps a team count needs before it can be used")
    ap.add_argument("--z", type=float, default=2.0, help="the gain must exceed z map-clustered standard errors")
    ap.add_argument("--max-width", type=int, default=0, help="0: infer capacity from layer shapes; explicit ceilings are enforced")
    ap.add_argument("--split", default="test")
    ap.add_argument("--fit-split", default="val")
    ap.add_argument("--out", required=True, help="output path without extension (.md and .json are written)")
    a = ap.parse_args(argv)
    layout = feature_layout()
    policy = export_weights.load_policy(Path(a.policy), layout)
    policy.calibrated = True
    value, beta = export_weights.load_value(Path(a.value), layout)
    if a.max_width < 0:
        raise SystemExit("--max-width must be positive, or 0 to infer capacity")
    a.max_width = a.max_width or export_weights.model_width(policy, value)
    try:
        nets = export_weights.quantize_model(policy, value, beta, layout, a.max_width)
    except ValueError as exc:
        raise SystemExit(f"cannot evaluate the exported model: {exc}") from exc
    tier = QuantizedTier(export_weights.flat_arrays(nets))
    counts = a.teams or served_counts(a.model) or model_counts(a.model)
    d_eval_all = load(a.shards, counts, a.split)
    d_fit_all = load(a.shards, counts, a.fit_split)
    qh_ok, qh_reason, baselines = baseline_status(a.shards, counts, [a.fit_split, a.split])
    index = load_index(a.shards)
    rows = lambda d: policy_rows(d, a.shards)  # real opponents only
    pick = lambda d, keep: {k: v[keep] for k, v in d.items()}
    keep_eval, keep_fit = rows(d_eval_all), rows(d_fit_all)
    d_eval = pick(d_eval_all, keep_eval)
    gate = {"min_gain": a.min_gain, "min_maps": a.min_maps, "z": a.z}
    pol = policy_section(policy, tier, pick(d_fit_all, keep_fit), d_eval, layout, a.adapter, gate, qh_ok, qh_reason,
                         map_clusters(d_eval, index))
    val = value_section(value, beta, d_eval_all, layout, tier)
    per_turn = 15 * export_weights.macs(nets["mover"]) + export_weights.macs(nets["squad"])
    sizes = {"params": {k: int(sum(l["w"].size + l["b"].size for l in q["layers"])) for k, q in nets.items()},
             "macs": {k: export_weights.macs(q) for k, q in nets.items()} | {"joint_policy_per_team": per_turn}}
    ids = lambda d: hashlib.sha256("\n".join(sorted({index["matches"][i]["match_id"] for i in np.unique(d["match"])}))
                                   .encode()).hexdigest()[:16]
    identity = export_weights.artifact_identity(a.model, a.policy, a.value, nets, layout, a.adapter) | {
        "dataset": {"team_counts": counts, "options": index["options"], "split": a.split, "fit_split": a.fit_split,
                    "eval_matches": ids(d_eval_all), "fit_matches": ids(d_fit_all)},
        "baseline": {"legacy_complete": qh_ok, "reason": qh_reason, "builds": baselines}}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(markdown(a.model, a.split, pol, val, sizes), encoding="utf-8")
    report = {"model": a.model, "team_counts": counts, "split": a.split, "fit_split": a.fit_split, "gate": gate,
              "identity": identity, "alpha": {str(k): v for k, v in pol["alpha"].items()}, "policy": pol, "value": val}
    out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(out.with_suffix(".md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
