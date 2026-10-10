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
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from fusion.models import ValueNet
from fusion.schema import layout as feature_layout
from training.value_cache import load_cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--value-family", choices=("v2", "v3"), required=True)
    ap.add_argument("--hidden", required=True, help="comma-separated hidden widths, e.g. 8192,512")
    ap.add_argument("--batches", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=20)
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
    import torch  # noqa: PLC0415

    from fusion import torch_backend as tb  # noqa: PLC0415
    dev = tb.pick_device(a.device)
    env = tb.configure(dev, a.tf32)
    L = feature_layout()
    train = load_cache(a.cache, "train", a.value_family)
    hidden = [int(x) for x in a.hidden.split(",")]
    results = []
    for batch in a.batches:
        rng = np.random.default_rng(a.seed)
        model = ValueNet(L, rng, hidden, a.value_family)
        args = SimpleNamespace(batch=batch, lr=a.lr, weight_decay=a.weight_decay, gpu_resident_gb=a.gpu_resident_gb,
                               gpu_window_rows=a.gpu_window_rows, rank_batch=0)
        row = {"batch": batch}
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            trainer = tb.ValueTrainer(model, L, train, train, {"aux_weight": a.aux_weight}, args, dev)
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
            seconds = time.perf_counter() - t0 if t0 else float("nan")
            row |= {"timed_steps": steps - a.warmup, "rows_per_second": round(rows / seconds) if seconds > 0 else None,
                    "seconds": round(seconds, 3), "train_outcome_loss": float(torch.stack(losses).mean()) if losses else None,
                    "resident": trainer.train.resident} | tb.peak_memory(dev)
        except torch.cuda.OutOfMemoryError as exc:  # recorded, not retried
            row |= {"error": f"out of memory: {str(exc).splitlines()[0]}"}
        results.append(row)
        print(f"[value_sweep] {row}", flush=True)
    params = int(sum(w.size + b.size for w, b in zip(model.net.W, model.net.b)))
    report = {"host": platform.node(), "env": env, "family": a.value_family, "hidden": hidden, "params": params,
              "rows": int(len(train["v"])), "steps": a.steps, "warmup": a.warmup, "seed": a.seed, "results": results,
              "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "note": "measured throughput and memory only; development-machine numbers are not evidence"}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
