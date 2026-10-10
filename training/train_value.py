"""Train the board value network V(s, team) for one model of the matrix (profiles/).

    python -m training.train_value --model general --out runs/value_general
    python -m training.train_value --model n08 --init runs/value_general --out runs/value_n08

Main target: the final pairwise outcome of the row's team (win 1, draw .5,
loss 0, averaged over opponents), the quantity the Bradley-Terry rating
rewards. Auxiliary targets (capture within 20 turns, own death within 10)
shape the hidden layer and are not exported. Every (turn, team) row is a
sample, whether or not that team's reply was valid.
"""
from __future__ import annotations

import argparse

import numpy as np

from fusion.models import ValueNet
from fusion.nn import Adam
from fusion.schema import layout as feature_layout
from fusion.paths import model_counts
from training.common import (Run, add_training_args, batches, data_manifest, fit, resolve_hidden, run_features, subset,
                             torch_fit, warm_start)
from training.dataset import load


def evaluate(model, layout, d, cfg, size=8192):
    total, wsum = 0.0, float(d["weight"].sum())
    for idx in batches(len(d["rec"]), size, None):
        w = d["weight"][idx]
        out = model.loss(model.inputs(d["rec"][idx], layout), d["v_y"][idx], w, cfg, backward=False)
        total += out["outcome"] * float(w.sum()) / wsum
    return total


def fit_value_temperature(z: np.ndarray, y: np.ndarray, w: np.ndarray, lo=0.05, hi=20.0, iters=100) -> float:
    """beta minimizing weighted BCE of sigmoid(beta z); convex, derivative sum w (sigmoid(beta z) - y) z."""
    def slope(b):
        return float((w * (1 / (1 + np.exp(-b * z)) - y) * z).sum())
    if slope(lo) >= 0:
        return lo
    if slope(hi) <= 0:
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if slope(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(ap, batch=2048)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--hidden", help="override preset value hidden widths, comma-separated")
    ap.add_argument("--value-family", choices=("v2", "v3"), default="v2",
                    help="input feature family (fusion/families.py): v2 legacy 54 features, v3 adds the extension")
    a = ap.parse_args(argv)
    hidden, capacity = resolve_hidden(a, "value")
    rng = np.random.default_rng(a.seed)
    fields = ("rec", "v_y", "weight", "teams", "match", "turn")
    counts = model_counts(a.model)
    train = load(a.shards, counts, "train", fields, families=(a.value_family,))
    val = load(a.shards, counts, a.val_split, fields, families=(a.value_family,))
    if a.count_weighting == "inv_teams":
        train["weight"] = (train["weight"] / train["teams"].astype(np.float32)).astype(np.float32)
        val["weight"] = (val["weight"] / val["teams"].astype(np.float32)).astype(np.float32)
    if a.max_rows and len(train["rec"]) > a.max_rows:
        train = subset(train, np.sort(rng.choice(len(train["rec"]), a.max_rows, replace=False)))
    layout = feature_layout()
    model = ValueNet(layout, rng, hidden, a.value_family)
    init = warm_start(model, a.init, layout.schema)
    opt = Adam(model.params, lr=a.lr, weight_decay=a.weight_decay)
    cfg = {"aux_weight": a.aux_weight}
    run = Run(a.out, {"kind": "value", "model": a.model, "capacity": capacity, "hidden": hidden, "args": vars(a), "seed": a.seed,
                      "init": init, "data": data_manifest(a.shards, counts),
                      "rows": {"train": len(train["rec"]), "val": len(val["rec"])}}
              | run_features(layout.schema, a.value_family))

    def train_epoch(r):
        total, wsum = 0.0, 0.0
        for idx in batches(len(train["rec"]), a.batch, r):
            w = train["weight"][idx]
            model.zero_grad()
            out = model.loss(model.inputs(train["rec"][idx], layout), train["v_y"][idx], w, cfg, backward=True)
            opt.step(model.grads)
            total += out["outcome"] * float(w.sum())
            wsum += float(w.sum())
        return total / wsum

    if a.backend == "torch":
        best = torch_fit(lambda tb, dev: tb.ValueTrainer(model, layout, train, val, cfg, a, dev), a, run, rng)
    else:
        best = fit(model, train_epoch, lambda: evaluate(model, layout, val, cfg), a, run, rng, opt)
    z = np.concatenate([model.logits(model.inputs(val["rec"][i], layout))[:, 0] for i in batches(len(val["rec"]), 8192, None)])
    beta = fit_value_temperature(z.astype(np.float64), val["v_y"][:, 0].astype(np.float64), val["weight"].astype(np.float64))
    run.log(stage="calibrated", beta=beta)
    run.save(model.state() | {"beta": np.array(beta)}, best_val=best, beta=beta)


if __name__ == "__main__":
    main()
