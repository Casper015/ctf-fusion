"""Train the board value network V(s, team) for one model of the matrix (profiles/).

    python -m training.train_value --model general --out runs/value_general
    python -m training.train_value --model n08 --init runs/value_general --out runs/value_n08
    python -m training.train_value --model general --cache data/vcache/v12xxl_v3 --value-family v3 \\
        --hidden 8192,512 --max-width 8192 --backend torch --device cuda --checkpoint --out runs/v_pilot_D_s1

Main target: the final pairwise outcome of the row's team (win 1, draw .5,
loss 0, averaged over opponents), the quantity the Bradley-Terry rating
rewards; on designated-carrier data (training.dataset --rows designated) it is
the expected pairwise utility under that carrier's continuation policy and the
schedule's opponent distribution, not the probability of finishing first.
Auxiliary targets (capture within 20 turns, own death within 10) shape the
hidden layer and are not exported. Every (turn, team) row is a sample, whether
or not that team's reply was valid. Early stopping and the calibration
temperature use the outcome loss on --val-split only.

Inputs: extractor shards (--shards) or a V-only cache (--cache, training.value_cache), which keeps only the
family's value columns, memory-mapped; the torch backend streams bounded windows of it to the device.

Optional ranking supervision (--rank DIR --rank-weight w): resolved candidate pairs from
training.branch_dataset add w x the pairwise logistic loss on the outcome logits of the two successor states,
P(a better than b) = sigmoid(z_a - z_b), weighted so every root counts once. Uncertain pairs (near ties) carry
no label and no weight. The outcome loss stays the primary objective and the stopping criterion.

Distillation (training.distill_value): --soft-targets FILE (a .npy aligned with the training rows, written by
distill_value with the teacher's calibrated probabilities) and --soft-weight w train the outcome head on
(1 - w) x real outcome + w x teacher probability; validation, stopping and calibration stay on real outcomes.

Semantics (unchanged, both backends): float32, clipped ReLU on [0, 1], weighted BCE summed over heads with
the aux weight, divided by the minibatch weight sum; Adam with coupled L2 decay on matrices only; minibatch
order from the run's seed. --checkpoint writes the best state (model_best.npz, best.json) on every
improvement, with the epoch and validation loss, so a long run keeps its provenance if it stops early.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fusion.models import ValueNet, value_columns
from fusion.nn import Adam
from fusion.paths import model_counts
from fusion.schema import layout as feature_layout
from training.common import (Run, add_training_args, batches, data_manifest, fit, resolve_hidden, run_features, subset,
                             torch_fit, warm_start)
from training.dataset import load

FIELDS = ("rec", "v_y", "weight", "teams", "match", "turn")


def rows_len(d: dict) -> int:
    return len(d["v"] if "v" in d else d["rec"])


def inputs(model: ValueNet, layout, d: dict, idx) -> np.ndarray:
    """Normalized value inputs of rows `idx`, from full records or from a cache's raw value columns."""
    if "v" in d:  # fusion.models.value_inputs on the family's columns: float64 division, then float32
        return (np.asarray(d["v"][idx]) / layout.value_scale(model.family)).astype(np.float32)
    return model.inputs(d["rec"][idx], layout)


def evaluate(model, layout, d, cfg, size=8192):
    total, wsum = 0.0, float(np.asarray(d["weight"]).sum())
    for idx in batches(rows_len(d), size, None):
        w = d["weight"][idx]
        out = model.loss(inputs(model, layout, d, idx), d["v_y"][idx], w, cfg, backward=False)
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


def load_rank(rank_dir: str, split: str, family: str, layout) -> dict | None:
    """Resolved candidate pairs of one split (training.branch_dataset), with the successors' value columns."""
    from fusion import families  # noqa: PLC0415
    path = Path(rank_dir) / split / "rank.npz"
    if not path.exists():
        return None
    index = json.loads((Path(rank_dir) / "index.json").read_text(encoding="utf-8"))
    families.check_shards(index["schema"], (family,), str(rank_dir))
    z = dict(np.load(path))
    keep = z["pair_y"] >= 0
    return {"cand_v": value_columns(z["cand_rec"], layout, family), "a": z["pair_a"][keep], "b": z["pair_b"][keep],
            "y": z["pair_y"][keep].astype(np.float32), "w": z["pair_w"][keep].astype(np.float32),
            "roots": int(len(z["root_teams"])), "pairs": int(len(keep)), "resolved": int(keep.sum())}


def load_data(a, counts):
    if a.cache:
        from training.value_cache import load_cache  # noqa: PLC0415
        meta = json.loads((Path(a.cache) / "meta.json").read_text(encoding="utf-8"))
        out = []
        for split in ("train", a.val_split):
            d = load_cache(a.cache, split, a.value_family)
            teams = np.asarray(d["teams"])
            if not set(np.unique(teams).tolist()) <= set(counts):  # this model serves fewer counts than the cache holds
                d = {k: np.asarray(v[np.isin(teams, counts)]) for k, v in d.items()}
            out.append(d)
        return out[0], out[1], {"cache": str(a.cache), "cache_meta": {k: meta[k] for k in ("features", "data", "rows")}}
    train = load(a.shards, counts, "train", FIELDS, families=(a.value_family,))
    val = load(a.shards, counts, a.val_split, FIELDS, families=(a.value_family,))
    return train, val, data_manifest(a.shards, counts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_training_args(ap, batch=2048)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--hidden", help="override preset value hidden widths, comma-separated (4XL: 8192,512 with --max-width 8192)")
    ap.add_argument("--value-family", choices=("v2", "v3"), default="v2",
                    help="input feature family (fusion/families.py): v2 legacy 54 features, v3 adds the extension")
    ap.add_argument("--cache", help="V-only cache from training.value_cache, instead of --shards")
    ap.add_argument("--rank", help="ranking pairs from training.branch_dataset (train split)")
    ap.add_argument("--rank-weight", type=float, default=0.0, help="weight of the pairwise ranking loss (0 = off)")
    ap.add_argument("--rank-batch", type=int, default=512, help="ranking pairs per optimizer step")
    ap.add_argument("--checkpoint", action="store_true", help="write the best state on every improvement")
    ap.add_argument("--soft-targets", help="teacher probabilities for the training rows (training.distill_value)")
    ap.add_argument("--soft-weight", type=float, default=0.0, help="share of the teacher's probability in the outcome target")
    a = ap.parse_args(argv)
    hidden, capacity = resolve_hidden(a, "value")
    rng = np.random.default_rng(a.seed)
    counts = model_counts(a.model)
    layout = feature_layout()
    train, val, data = load_data(a, counts)
    if a.count_weighting == "inv_teams":
        train["weight"] = (np.asarray(train["weight"]) / np.asarray(train["teams"]).astype(np.float32)).astype(np.float32)
        val["weight"] = (np.asarray(val["weight"]) / np.asarray(val["teams"]).astype(np.float32)).astype(np.float32)
    soft = None
    if a.soft_targets and a.soft_weight > 0:
        soft = np.load(a.soft_targets)
        meta = json.loads(Path(a.soft_targets).with_suffix(".json").read_text(encoding="utf-8"))
        if len(soft) != rows_len(train) or meta.get("cache") != str(a.cache):
            raise SystemExit(f"{a.soft_targets} does not belong to these training rows (made for {meta.get('cache')}, "
                             f"{len(soft)} rows; here {a.cache}, {rows_len(train)} rows)")
        y = np.array(train["v_y"], dtype=np.float32)
        y[:, 0] = (1 - a.soft_weight) * y[:, 0] + a.soft_weight * soft.astype(np.float32)
        train = dict(train) | {"v_y": y}
        data = data | {"soft_targets": meta | {"path": str(a.soft_targets), "weight": a.soft_weight}}
    if a.max_rows and rows_len(train) > a.max_rows:
        train = subset(train, np.sort(rng.choice(rows_len(train), a.max_rows, replace=False)))
    rank = load_rank(a.rank, "train", a.value_family, layout) if a.rank and a.rank_weight > 0 else None
    if a.rank and a.rank_weight > 0 and rank is None:
        raise SystemExit(f"{a.rank} has no train split")
    model = ValueNet(layout, rng, hidden, a.value_family)
    init = warm_start(model, a.init, layout.schema)
    opt = Adam(model.params, lr=a.lr, weight_decay=a.weight_decay)
    cfg = {"aux_weight": a.aux_weight, "rank_weight": a.rank_weight}
    params = int(sum(w.size + b.size for w, b in zip(model.net.W, model.net.b)))
    run = Run(a.out, {"kind": "value", "model": a.model, "capacity": capacity, "hidden": hidden, "args": vars(a), "seed": a.seed,
                      "init": init, "data": data, "params": params,
                      "rows": {"train": rows_len(train), "val": rows_len(val)},
                      "rank": None if rank is None else {k: rank[k] for k in ("roots", "pairs", "resolved")} | {"dir": a.rank}}
              | run_features(layout.schema, a.value_family))
    started = time.time()

    def on_best(epoch, val_loss, state):
        if a.checkpoint:
            np.savez(Path(a.out) / "model_best.npz", **state)
            (Path(a.out) / "best.json").write_text(json.dumps({"epoch": epoch, "val": val_loss, "seconds": round(time.time() - started, 1),
                                                               "config": run.config}, indent=1, default=str), encoding="utf-8")

    rank_x = None if rank is None else (rank["cand_v"] / layout.value_scale(a.value_family)).astype(np.float32)
    cursor = {"perm": None, "pos": 0}

    def rank_step(r):
        n = len(rank["y"])
        if cursor["perm"] is None or cursor["pos"] >= n:
            cursor["perm"], cursor["pos"] = r.permutation(n), 0
        p = cursor["perm"][cursor["pos"]:cursor["pos"] + a.rank_batch]
        cursor["pos"] += a.rank_batch
        return model.rank_loss(rank_x[rank["a"][p]], rank_x[rank["b"][p]], rank["y"][p], rank["w"][p], cfg, backward=True)

    def train_epoch(r):
        total, wsum = 0.0, 0.0
        for idx in batches(rows_len(train), a.batch, r):
            w = train["weight"][idx]
            model.zero_grad()
            out = model.loss(inputs(model, layout, train, idx), train["v_y"][idx], w, cfg, backward=True)
            if rank is not None:
                rank_step(r)
            opt.step(model.grads)
            total += out["outcome"] * float(w.sum())
            wsum += float(w.sum())
        return total / wsum

    if a.backend == "torch":
        best = torch_fit(lambda tb, dev: tb.ValueTrainer(model, layout, train, val, cfg, a, dev, rank), a, run, rng, on_best)
    else:
        best = fit(model, train_epoch, lambda: evaluate(model, layout, val, cfg), a, run, rng, opt, on_best)
    z = np.concatenate([model.logits(inputs(model, layout, val, i))[:, 0] for i in batches(rows_len(val), 8192, None)])
    beta = fit_value_temperature(z.astype(np.float64), np.asarray(val["v_y"])[:, 0].astype(np.float64),
                                 np.asarray(val["weight"]).astype(np.float64))
    seconds = round(time.time() - started, 1)
    run.log(stage="calibrated", beta=beta, seconds=seconds)
    run.save(model.state() | {"beta": np.array(beta)}, best_val=best, beta=beta, train_seconds=seconds)


if __name__ == "__main__":
    main()
