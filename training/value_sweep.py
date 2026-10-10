"""Batch-size sweep for a value network shape on the torch backend: throughput and device memory, measured.

    python tools/repo/task.py lock gpu --task T00xx -- python -m training.value_sweep --cache data/vcache/v12xxl_v3 \\
        --value-family v3 --hidden 8192,512 --batches 2048 4096 8192 --steps 300 --device cuda \\
        --out reports/models/v_pilot/sweep_DESKTOP-G60OJUD.json

For each batch size: a fresh model from --seed (the trainer's float32 semantics: fusion.torch_backend.ValueTrainer,
TF32 off unless --tf32), --warmup optimizer steps, then --steps timed steps over the training rows (resident on
the device when they fit --gpu-resident-gb, else streamed in windows of --gpu-window-rows). Recorded per batch:
rows per second over the timed steps, peak allocated and reserved device memory (reset before each batch), the
mean training loss of the timed steps, and whether the rows were resident. An out-of-memory error is recorded
as such, not retried. Nothing here is a training result; it picks a batch size and shows what it costs.
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fusion.models import ValueNet
from fusion.schema import layout as feature_layout
from training.value_cache import cache_identity, load_cache


def measure_batch(a, L, train, val, hidden, dev, batch):
    """Scoped device objects: release the entire preceding trial before measuring the next."""
    import torch
    from fusion import torch_backend as tb
    rng = np.random.default_rng(a.seed)
    model = ValueNet(L, rng, hidden, a.value_family)
    args = SimpleNamespace(batch=batch, val_batch=a.val_batch or batch, lr=a.lr, weight_decay=a.weight_decay,
                           gpu_resident_gb=a.gpu_resident_gb, gpu_window_rows=a.gpu_window_rows, rank_batch=0)
    trainer = tb.ValueTrainer(model, L, train, val, {"aux_weight": a.aux_weight}, args, dev)
    order = rng.permutation(trainer.train.n)
    losses, steps, t0, rows = [], 0, None, 0
    for b in trainer.train.minibatches(np.resize(order, (a.warmup + a.steps) * batch), batch):
        if steps == a.warmup:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            t0 = time.perf_counter()
        trainer.opt.zero_grad()
        loss, out = trainer.net.loss(trainer._x(b["v"]), b["v_y"], b["weight"], trainer.cfg)
        loss.backward()
        trainer.opt.step()
        if steps >= a.warmup:
            losses.append(out["outcome"])
            rows += len(b["weight"])
        steps += 1
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    seconds = time.perf_counter() - t0
    tval = time.perf_counter()
    validation = trainer.evaluate()
    return {"timed_steps": steps - a.warmup, "optimizer_steps": trainer.opt.t, "rows_per_second": round(rows / seconds),
            "seconds": round(seconds, 3), "validation_seconds": round(time.perf_counter() - tval, 3),
            "validation_outcome_loss": validation, "validation_rows": trainer.val.n, "val_batch": args.val_batch,
            "train_outcome_loss": float(torch.stack(losses).mean()), "resident": trainer.train.resident,
            "validation_resident": trainer.val.resident} | tb.peak_memory(dev)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--value-family", choices=("v2", "v3"), required=True)
    ap.add_argument("--hidden", required=True, help="comma-separated hidden widths, e.g. 8192,512")
    ap.add_argument("--batches", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--val-batch", type=int, default=0, help="validation batch (0 = the training trial's batch)")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--aux-weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--gpu-resident-gb", type=float, default=6.0)
    ap.add_argument("--gpu-window-rows", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.steps < 1 or a.warmup < 0 or a.val_batch < 0 or any(b < 1 for b in a.batches):
        raise SystemExit("steps and batches must be positive; warmup and val-batch must be nonnegative")
    import torch  # noqa: PLC0415

    from fusion import torch_backend as tb  # noqa: PLC0415
    dev = tb.pick_device(a.device)
    env = tb.configure(dev, a.tf32)
    L = feature_layout()
    train = load_cache(a.cache, "train", a.value_family)
    val = load_cache(a.cache, a.val_split, a.value_family)
    hidden = [int(x) for x in a.hidden.split(",")]
    results = []
    for batch in a.batches:
        row = {"batch": batch}
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            row |= measure_batch(a, L, train, val, hidden, dev, batch)
        except torch.cuda.OutOfMemoryError as exc:  # recorded, not retried
            row |= {"error": f"out of memory: {str(exc).splitlines()[0]}"} | tb.peak_memory(dev)
        finally:
            gc.collect()
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        results.append(row)
        print(f"[value_sweep] {row}", flush=True)
    sizes = [len(L.value_scale(a.value_family)), *hidden, 3]
    params = sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:]))
    report = {"host": platform.node(), "env": env, "family": a.value_family, "hidden": hidden, "params": params,
              "rows": int(len(train["v"])), "steps": a.steps, "warmup": a.warmup, "seed": a.seed, "results": results,
              "validation_split": a.val_split, "cache_meta": json.loads((Path(a.cache) / "meta.json").read_text(encoding="utf-8")),
              "cache_identity": cache_identity(a.cache),
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "note": "measured throughput and memory only; development-machine numbers are not evidence"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
