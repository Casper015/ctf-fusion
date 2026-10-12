"""Train a small own-action scorer Q_theta(history, candidate) on incumbent-relative gains (task T0034).

    python -m training.train_decision_ranker --data data/decision/t0010_p2 --model linear --exploratory \\
        --seed 1 --out runs/decision_linear_s1
    python -m training.train_decision_ranker --data data/decision/t0010_p2 --model mlp --hidden 64,16 --exploratory \\
        --seed 1 --out runs/decision_mlp_s1

Target (exploratory): A^pi0(s, a) = E[U_T | s, a, then pi0] - E[U_T | s, a0, then pi0], estimated per root by the
difference of mean continuation outcomes of candidate a and the incumbent a0 (training.decision_dataset). The scorer
predicts g(a) = Q(s, a) - Q(s, a0), so g(a0) = 0 exactly and only differences between complete orders are learned.

Models (start simple):
  linear  Q = w . phi(a): the candidate features only (a root-shared term cancels in g);
  mlp     Q = MLP([psi(s), phi(a)]) with clipped ReLU hidden layers: state-action interactions.
psi(s) and phi(a) are fusion.decision_features.root_features / candidate_features of the root record and the
canonical order; no outcome, later block or realized successor is an input.

Losses: --loss gain (default) weighted squared error of g against the observed gain; --loss pairwise a logistic
loss on within-root preferences whose outcome gap exceeds one pairwise step (and two standard errors when repeats
allow). Weights: every training root with labels counts equally within its team count, and team counts count
equally; inside a root, items share its weight. With repeats, an item's weight is scaled by the inverse variance
of its observed gain (sample variances, or the pooled within-action variance when one side has a single repeat);
with one repeat everywhere no variance exists and the weighting is uniform (recorded in the run config).

Early stopping on the validation loss only; the test split is never opened here. The run's config records kind
own_action_scorer: export_weights/export_mixed refuse it as opponent P or V, and this loader refuses any other
kind. The scorer reads the opponent P nowhere; the carrier configuration (P identity, search, budget) of the data
is copied from the dataset index so a later deployment can be checked against it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from fusion.nn import MLP, Adam
from training import decision_dataset as D
from training.common import Run
from training.evaluate_decisions import count_weights, mask_gains

VAR_FLOOR = 1e-4


class Scorer:
    def __init__(self, model: str, root_dim: int, cand_dim: int, hidden=(), rng=None):
        if model not in ("linear", "mlp"):
            raise SystemExit(f"unknown scorer model {model}")
        self.model, self.root_dim, self.cand_dim, self.hidden = model, root_dim, cand_dim, tuple(hidden)
        rng = rng or np.random.default_rng(0)
        sizes = [cand_dim, 1] if model == "linear" else [root_dim + cand_dim, *self.hidden, 1]
        self.net = MLP(sizes, rng)

    def inputs(self, xr: np.ndarray, xc: np.ndarray) -> np.ndarray:
        return xc if self.model == "linear" else np.concatenate([xr, xc], axis=1)

    def q(self, xr, xc, keep=False) -> np.ndarray:
        return self.net.forward(self.inputs(xr, xc).astype(np.float32), keep=keep)[:, 0]

    def gains(self, d: dict) -> np.ndarray:
        """g per candidate: Q(c) - Q(incumbent of its root); NaN outside the support, 0 at the incumbent."""
        xr, xc = D.scorer_inputs(d)
        q = self.q(xr, xc) if len(xc) else np.zeros(0)
        inc = d["root_incumbent"][d["cand_root"]]
        return mask_gains(d, (q - q[inc]).astype(np.float64))

    def state(self) -> dict:
        return self.net.state() | {"kind": np.array(F.KIND), "model": np.array(self.model),
                                   "hidden": np.array(self.hidden, np.int64)}


def load_scorer(run: Path, index: dict | None = None) -> Scorer:
    """An own-action scorer run; refuses P, V or any other kind, and a feature identity the extractor lacks."""
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    if cfg.get("kind") != F.KIND:
        raise SystemExit(f"{run}: kind {cfg.get('kind')!r} is not an own-action scorer")
    from fusion.schema import layout  # noqa: PLC0415
    now = F.feature_identity(layout())
    if cfg["features"] != now:
        raise SystemExit(f"{run}: trained on features {cfg['features']}, the extractor provides {now}")
    if index is not None and index["features"] != cfg["features"]:
        raise SystemExit(f"{run}: the dataset's features {index['features']} differ from the scorer's")
    st = dict(np.load(Path(run) / "model.npz"))
    if str(st.get("kind")) != F.KIND:
        raise SystemExit(f"{run}: model.npz is not an own-action scorer")
    s = Scorer(str(st["model"]), cfg["root_dim"], cfg["cand_dim"], tuple(int(h) for h in st["hidden"]))
    s.net.load(st)
    return s


def pooled_within_variance(d: dict) -> float | None:
    """Pooled sample variance of repeats of the same candidate; None when no candidate has two repeats."""
    n, v = d["cand_n"].astype(np.float64), d["cand_var"]
    m = (n >= 2) & np.isfinite(v)
    if not m.any():
        return None
    return float(((n[m] - 1) * v[m]).sum() / (n[m] - 1).sum())


def gain_items(d: dict, pooled: float | None) -> dict:
    """Training items: (candidate, incumbent, observed gain, weight), plus how uncertainty entered."""
    cand, inc_of, target, var = [], [], [], []
    root_items = {}
    for r, (first, count) in enumerate(zip(d["root_first"], d["root_count"])):
        inc = int(d["root_incumbent"][r])
        if d["cand_n"][inc] == 0 or not d["cand_support"][inc]:
            continue
        for c in range(first, first + count):
            if c == inc or not d["cand_support"][c] or d["cand_n"][c] == 0:
                continue
            cand.append(c)
            inc_of.append(inc)
            target.append(d["cand_y"][c] - d["cand_y"][inc])
            vc, v0 = d["cand_var"][c], d["cand_var"][inc]
            nc, n0 = d["cand_n"][c], d["cand_n"][inc]
            if np.isfinite(vc) and np.isfinite(v0):
                var.append(vc / nc + v0 / n0)
            elif pooled is not None:
                var.append(pooled * (1 / nc + 1 / n0))
            else:
                var.append(np.nan)
            root_items.setdefault(r, []).append(len(cand) - 1)
    cand, inc_of, target, var = map(np.asarray, (cand, inc_of, target, var))
    w = np.zeros(len(cand))
    roots = np.array(sorted(root_items))
    if len(roots):
        rw = count_weights(d["root_teams"][roots])
        for r, wr in zip(roots, rw):
            items = root_items[r]
            w[items] = wr / len(items)
    mode = "none (one repeat: no variance estimate)"
    if len(var) and np.isfinite(var).all():
        prec = 1.0 / (var + VAR_FLOOR)
        w = w * prec / np.average(prec, weights=np.maximum(w, 1e-300))
        mode = "inverse variance of the observed gain" + (" (pooled within-action variance where needed)" if pooled else "")
    elif len(var) and np.isfinite(var).any():
        mode = "none (variance available for only some items)"
    return {"cand": cand.astype(np.int64), "inc": inc_of.astype(np.int64), "y": target.astype(np.float64),
            "w": w / w.sum() if w.sum() > 0 else w, "uncertainty": mode}


def pair_items(d: dict) -> dict:
    """Within-root preferences: (a, b, a better) when the gap exceeds one pairwise step (and 2 SE with repeats)."""
    a, b, w = [], [], []
    root_pairs = {}
    for r, (first, count) in enumerate(zip(d["root_first"], d["root_count"])):
        if not d["cand_support"][d["root_incumbent"][r]]:
            continue
        idx = [c for c in range(first, first + count) if d["cand_support"][c] and d["cand_n"][c] > 0]
        step = 0.5 / max(int(d["root_teams"][r]) - 1, 1)
        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                ci, cj = idx[i], idx[j]
                diff = d["cand_y"][ci] - d["cand_y"][cj]
                se = np.sqrt(d["cand_var"][ci] / d["cand_n"][ci] + d["cand_var"][cj] / d["cand_n"][cj])
                if abs(diff) <= step or (np.isfinite(se) and abs(diff) <= 2 * se):
                    continue
                hi, lo = (ci, cj) if diff > 0 else (cj, ci)
                a.append(hi)
                b.append(lo)
                root_pairs.setdefault(r, []).append(len(a) - 1)
    w = np.zeros(len(a))
    roots = np.array(sorted(root_pairs))
    if len(roots):
        for r, wr in zip(roots, count_weights(d["root_teams"][roots])):
            w[root_pairs[r]] = wr / len(root_pairs[r])
    return {"a": np.asarray(a, np.int64), "b": np.asarray(b, np.int64), "w": w}


def loss_and_grad(scorer: Scorer, xr, xc, items: dict, loss: str, backward: bool):
    q = scorer.q(xr, xc, keep=backward)
    dq = np.zeros_like(q, dtype=np.float64)
    if loss == "gain":
        g = q[items["cand"]] - q[items["inc"]]
        e = g - items["y"]
        val = float((items["w"] * e * e).sum())
        if backward:
            np.add.at(dq, items["cand"], 2 * items["w"] * e)
            np.add.at(dq, items["inc"], -2 * items["w"] * e)
    else:
        z = q[items["a"]] - q[items["b"]]
        val = float((items["w"] * np.logaddexp(0.0, -z)).sum())
        if backward:
            s = -items["w"] / (1.0 + np.exp(z))
            np.add.at(dq, items["a"], s)
            np.add.at(dq, items["b"], -s)
    if backward:
        scorer.net.zero_grad()
        scorer.net.backward(dq[:, None].astype(np.float32))
    return val


def subset_items(items: dict, sel: np.ndarray, loss: str) -> dict:
    keys = ("cand", "inc", "y", "w") if loss == "gain" else ("a", "b", "w")
    out = {k: items[k][sel] for k in keys}
    s = out["w"].sum()
    out["w"] = out["w"] / s if s > 0 else out["w"]
    return out


def train(a) -> dict:
    data = Path(a.data)
    index = D.index_of(data)
    if not index["authoritative"] and not a.exploratory:
        raise SystemExit("exploratory dataset (" + "; ".join(index["exploratory_reasons"]) + "): pass --exploratory to train on it")
    D.check_isolation(data)
    tr, va = D.load(data, "train"), D.load(data, "val")
    if set(tr["root_map"].tolist()) & set(va["root_map"].tolist()):
        raise SystemExit("train and val share maps")
    from fusion.schema import layout  # noqa: PLC0415
    L = layout()
    pooled = pooled_within_variance(tr)
    make = (lambda d: gain_items(d, pooled)) if a.loss == "gain" else pair_items
    it_tr, it_va = make(tr), make(va)
    n_items = len(it_tr["w"])
    if n_items == 0:
        raise SystemExit("no training items: no root has a supported, labeled alternative to its incumbent")
    xr_tr, xc_tr = D.scorer_inputs(tr, L)
    xr_va, xc_va = D.scorer_inputs(va, L)
    rng = np.random.default_rng(a.seed)
    hidden = tuple(int(h) for h in a.hidden.split(",")) if a.model == "mlp" and a.hidden else ()
    scorer = Scorer(a.model, F.root_dim(L), F.CAND_DIM, hidden, rng)
    opt = Adam(scorer.net.params, lr=a.lr, weight_decay=a.weight_decay)
    cfg = {"kind": F.KIND, "model": a.model, "hidden": list(hidden), "loss": a.loss, "features": F.feature_identity(L),
           "root_dim": F.root_dim(L), "cand_dim": F.CAND_DIM, "args": vars(a), "seed": a.seed,
           "dataset": {"path": str(data), "index_sha256": D.sha256_file(data / "index.json"),
                       "train_sha256": index["splits"]["train"]["sha256"], "val_sha256": index["splits"]["val"]["sha256"],
                       "authoritative": index["authoritative"], "exploratory_reasons": index["exploratory_reasons"]},
           "carrier": index["carrier"], "uncertainty": it_tr.get("uncertainty", "pairwise: gaps over one step / 2 SE"),
           "items": {"train": n_items, "val": int(len(it_va["w"]))},
           "opponent_policy": "frozen: not read or trained by the scorer"}
    run = Run(a.out, cfg)
    best, best_state, bad = np.inf, scorer.state(), 0
    order = np.arange(n_items)
    for epoch in range(1, a.epochs + 1):
        rng.shuffle(order)
        for start in range(0, n_items, a.batch):
            sel = order[start:start + a.batch]
            loss_and_grad(scorer, xr_tr, xc_tr, subset_items(it_tr, sel, a.loss), a.loss, True)
            opt.step(scorer.net.grads)
        tl = loss_and_grad(scorer, xr_tr, xc_tr, it_tr, a.loss, False)
        vl = loss_and_grad(scorer, xr_va, xc_va, it_va, a.loss, False) if len(it_va["w"]) else float("nan")
        if not np.isfinite(tl):
            raise SystemExit(f"non-finite training loss at epoch {epoch}")
        run.log(epoch=epoch, train=tl, val=vl)
        score = vl if np.isfinite(vl) else tl
        if score < best - 1e-12:
            best, best_state, bad = score, {k: np.copy(v) for k, v in scorer.state().items()}, 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    scorer.net.load(best_state)
    run.save(scorer.state(), best_val=best)
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=("linear", "mlp"), default="linear")
    ap.add_argument("--hidden", default="64,16", help="mlp hidden widths")
    ap.add_argument("--loss", choices=("gain", "pairwise"), default="gain")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--exploratory", action="store_true", help="allow a dataset that is not authoritative (recorded)")
    ap.add_argument("--out", required=True)
    train(ap.parse_args(argv))


if __name__ == "__main__":
    main()
