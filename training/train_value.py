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

from fusion.models import ValueNet, value_inputs
from fusion.nn import Adam
from fusion.schema import layout as feature_layout
from fusion.paths import model_counts
from training.common import Run, add_training_args, batches, check_widths, data_manifest, fit, subset, warm_start
from training.dataset import load


def evaluate(model, layout, d, cfg, size=8192):
    total, wsum = 0.0, float(d["weight"].sum())
    for idx in batches(len(d["rec"]), size, None):
        w = d["weight"][idx]
        out = model.loss(value_inputs(d["rec"][idx], layout), d["v_y"][idx], w, cfg, backward=False)
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
    ap.add_argument("--hidden", default="128,16")
    a = ap.parse_args(argv)
    hidden = check_widths(a.hidden.split(","), "--hidden")
    rng = np.random.default_rng(a.seed)
    fields = ("rec", "v_y", "weight", "teams", "match", "turn")
    counts = model_counts(a.model)
    train, val = load(a.shards, counts, "train", fields), load(a.shards, counts, a.val_split, fields)
    if a.max_rows and len(train["rec"]) > a.max_rows:
        train = subset(train, np.sort(rng.choice(len(train["rec"]), a.max_rows, replace=False)))
    layout = feature_layout()
    model = ValueNet(layout, rng, hidden)
    init = warm_start(model, a.init, layout.schema["spec_version"])
    opt = Adam(model.params, lr=a.lr, weight_decay=a.weight_decay)
    cfg = {"aux_weight": a.aux_weight}
    run = Run(a.out, {"kind": "value", "model": a.model, "hidden": hidden, "args": vars(a), "seed": a.seed,
                      "init": init, "data": data_manifest(a.shards, counts),
                      "rows": {"train": len(train["rec"]), "val": len(val["rec"])},
                      "spec_version": layout.schema["spec_version"]})

    def train_epoch(r):
        total, wsum = 0.0, 0.0
        for idx in batches(len(train["rec"]), a.batch, r):
            w = train["weight"][idx]
            model.zero_grad()
            out = model.loss(value_inputs(train["rec"][idx], layout), train["v_y"][idx], w, cfg, backward=True)
            opt.step(model.grads)
            total += out["outcome"] * float(w.sum())
            wsum += float(w.sum())
        return total / wsum

    best = fit(model, train_epoch, lambda: evaluate(model, layout, val, cfg), a, run, rng, opt)
    z = np.concatenate([model.logits(value_inputs(val["rec"][i], layout))[:, 0] for i in batches(len(val["rec"]), 8192, None)])
    beta = fit_value_temperature(z.astype(np.float64), val["v_y"][:, 0].astype(np.float64), val["weight"].astype(np.float64))
    run.log(stage="calibrated", beta=beta)
    run.save(model.state() | {"beta": np.array(beta)}, best_val=best, beta=beta)


if __name__ == "__main__":
    main()
