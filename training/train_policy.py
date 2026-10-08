"""Train the opponent squad policy q_theta(b|s) for one model of the matrix (profiles/).

    python -m training.train_policy --model general --out runs/policy_general
    python -m training.train_policy --model n08 --init runs/policy_general --out runs/policy_n08

Supervised on the requested joint moves (125-way, masked to legal joint
moves), the conditional action head and the attack-target head. After
training, one inverse temperature per head is fitted on the validation split
(convex, exact) and folded into the export.
"""
from __future__ import annotations

import argparse

import numpy as np

from fusion.mixture import fit_temperature
from fusion.models import PolicyNet, policy_batch
from fusion.nn import Adam
from fusion.schema import layout as feature_layout
from fusion.paths import model_counts
from training.common import Run, add_training_args, batches, data_manifest, fit, policy_rows, resolve_hidden, subset, warm_start
from training.dataset import load

LABELS = ("y_move", "y_joint", "y_act", "y_tgt")


def prepare(shards, counts, split, max_rows, rng, keep_unlabeled=False):
    """Policy rows of a split: valid replies from families that give labels (not the noise bot)."""
    d = load(shards, counts, split)
    keep = policy_rows(d, shards, keep_unlabeled)
    idx = np.nonzero(keep)[0]
    if max_rows and len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    return subset(d, idx)


def check_labels(d, layout):
    """Every canonical label must sit on a legal entry, or training would silently ignore it."""
    legal = layout.get(d["rec"], "legal").astype(bool)
    k = np.arange(3)[None, :]
    r = np.arange(len(d["rec"]))[:, None]
    assert legal[r, k, d["y_move"]].all(), "move label on an illegal move"


def evaluate(model, layout, d, cfg, size=4096):
    total, parts = 0.0, {"joint": 0.0, "act": 0.0, "target": 0.0}
    wsum = float(d["weight"].sum())
    for idx in batches(len(d["rec"]), size, None):
        X = policy_batch(d["rec"][idx], layout)
        w = d["weight"][idx]
        out = model.loss(X, {k: d[k][idx] for k in LABELS}, w, cfg, backward=False)
        share = float(w.sum()) / wsum
        total += out["total"] * share
        for k in parts:
            parts[k] += out[k] * share
    return total, parts


def collect_logits(model, layout, d, size=4096):
    joint, act, tgt = [], [], []
    for idx in batches(len(d["rec"]), size, None):
        X = policy_batch(d["rec"][idx], layout)
        move = d["y_move"][idx].astype(np.int64)
        joint.append(model.joint_logits(X))
        act.append((model.act_logits(X, move).reshape(-1, 4), X.act_mask(move).reshape(-1, 4)))
        zt = model.target_logits(X, move)
        tgt.append((zt.reshape(-1, zt.shape[-1]), X.cand_valid.reshape(-1, zt.shape[-1])))
    return joint, act, tgt


def calibrate(model, layout, d):
    joint, act, tgt = collect_logits(model, layout, d)
    w = d["weight"].astype(np.float64)
    mask = np.concatenate([policy_batch(d["rec"][i], layout).joint_mask for i in batches(len(d["rec"]), 4096, None)])
    model.beta["joint"] = fit_temperature(np.concatenate(joint), mask, d["y_joint"].astype(np.int64), w)
    w3 = np.repeat(w, 3)
    model.beta["act"] = fit_temperature(np.concatenate([a for a, _ in act]), np.concatenate([m for _, m in act]),
                                        d["y_act"].reshape(-1).astype(np.int64), w3)
    model.beta["target"] = fit_temperature(np.concatenate([a for a, _ in tgt]), np.concatenate([m for _, m in tgt]),
                                           d["y_tgt"].reshape(-1).astype(np.int64), w3)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(ap, batch=1024)
    ap.add_argument("--smoothing", type=float, default=0.01)
    ap.add_argument("--act-weight", type=float, default=0.5)
    ap.add_argument("--target-weight", type=float, default=0.5)
    ap.add_argument("--hidden-squad", help="override preset squad hidden widths, comma-separated")
    ap.add_argument("--hidden-mover", help="override preset mover hidden widths, comma-separated")
    ap.add_argument("--hidden-act", help="override preset action hidden widths, comma-separated")
    ap.add_argument("--hidden-target", help="override preset target hidden widths, comma-separated")
    ap.add_argument("--keep-unlabeled", action="store_true",
                    help="also learn from families marked policy_labels false in pool.json (the noise bot)")
    a = ap.parse_args(argv)
    hidden, capacity = resolve_hidden(a, "policy")
    rng = np.random.default_rng(a.seed)
    counts = model_counts(a.model)
    train = prepare(a.shards, counts, "train", a.max_rows, rng, a.keep_unlabeled)
    val = prepare(a.shards, counts, a.val_split, 0, rng, a.keep_unlabeled)
    layout = feature_layout()
    check_labels(train, layout)
    model = PolicyNet(layout, rng, hidden)
    init = warm_start(model, a.init, layout.schema["spec_version"])
    opt = Adam(model.params, lr=a.lr, weight_decay=a.weight_decay)
    cfg = {"smoothing": a.smoothing, "act_weight": a.act_weight, "target_weight": a.target_weight}
    run = Run(a.out, {"kind": "policy", "model": a.model, "capacity": capacity, "hidden": hidden, "args": vars(a), "seed": a.seed,
                      "init": init, "data": data_manifest(a.shards, counts),
                      "rows": {"train": len(train["rec"]), "val": len(val["rec"])},
                      "spec_version": layout.schema["spec_version"]})

    def train_epoch(r):
        total, wsum = 0.0, 0.0
        for idx in batches(len(train["rec"]), a.batch, r):
            X = policy_batch(train["rec"][idx], layout)
            w = train["weight"][idx]
            model.zero_grad()
            out = model.loss(X, {k: train[k][idx] for k in LABELS}, w, cfg, backward=True)
            opt.step(model.grads)
            total += out["total"] * float(w.sum())
            wsum += float(w.sum())
        return total / wsum

    best = fit(model, train_epoch, lambda: evaluate(model, layout, val, cfg)[0], a, run, rng, opt)
    calibrate(model, layout, val)
    model.calibrated = True
    _, parts = evaluate(model, layout, val, cfg | {"smoothing": 0.0})
    run.log(stage="calibrated", **{f"val_{k}": v for k, v in parts.items()}, **{f"beta_{k}": v for k, v in model.beta.items()})
    run.save(model.state(), best_val=best, beta=model.beta)


if __name__ == "__main__":
    main()
