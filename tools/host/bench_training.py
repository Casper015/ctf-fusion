"""Measure training throughput per backend before relying on the GPU (host tool; writes a report, trains nothing kept).

    python tools/host/bench_training.py --model n15 --tier XXL --max-rows 200000
    python tools/host/bench_training.py --model general --tier S XXL --backends numpy torch-cuda --batches 1024 4096

For each (tier, backend, batch) it trains the policy and value nets for --epochs on the same --max-rows rows
(same seed, so the same rows and minibatch order) and records the wall time of the trainer call, epoch time,
rows per second, the validation loss after training (quality at equal epochs, not at equal time) and, on CUDA,
peak allocated and reserved memory. Startup (loading shards, building the model) is included in the wall time
and reported apart from the epoch time. Larger batches mean fewer updates per epoch: compare their validation
loss too, not only their speed.

Output: reports/host/training_bench_<host>.md and .json. Backends: numpy (the reference), torch-cpu, torch-cuda.
"""
from __future__ import annotations

import argparse
import json
import platform
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "host"))

from fusion.capacity import TIERS  # noqa: E402
from training import train_policy, train_value  # noqa: E402


def run_one(kind: str, tier: str, backend: str, batch: int, a, tmp: Path) -> dict:
    out = tmp / f"{kind}_{tier}_{backend}_{batch}"
    common = ["--shards", a.shards, "--model", a.model, "--epochs", str(a.epochs), "--batch", str(batch),
              "--max-rows", str(a.max_rows), "--seed", "1", "--out", str(out), "--lr-schedule", "constant",
              "--capacity", tier]
    if backend != "numpy":
        common += ["--backend", "torch", "--device", backend.split("-", 1)[1]] + (["--tf32"] if a.tf32 else [])
    args, trainer = common, train_policy if kind == "policy" else train_value
    t0 = time.time()
    trainer.main(args)
    wall = time.time() - t0
    log = [json.loads(line) for line in (out / "log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    epochs = [r for r in log if "epoch" in r]
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    rows = cfg["rows"]["train"]
    epoch_s = sum(r["seconds"] for r in epochs)
    be = cfg.get("backend", {})
    return {"kind": kind, "tier": tier, "backend": backend, "batch": batch, "rows": rows, "epochs": len(epochs),
            "wall_seconds": round(wall, 2), "epoch_seconds": round(epoch_s, 3),
            "rows_per_second": round(rows * len(epochs) / epoch_s) if epoch_s >= 0.05 else None,   # too short to time
            "val": epochs[-1]["val"] if epochs else None,
            "gpu": be.get("gpu"), "peak_allocated_gb": be.get("peak_allocated_gb"), "peak_reserved_gb": be.get("peak_reserved_gb")}


def markdown(rows: list[dict], meta: dict) -> str:
    out = [f"# Training throughput on {meta['host']}", "",
           f"Model `{meta['model']}`, {meta['max_rows']} training rows, {meta['epochs']} epoch(s), constant learning rate, seed 1. "
           f"{meta['platform']}; {meta.get('gpu') or 'no GPU used'}. Epoch time excludes loading; wall time includes it. "
           "Validation loss is after the same number of epochs (larger batches take fewer steps).", "",
           "| Net | Tier | Backend | Batch | Rows/s | Epoch s | Wall s | Speedup vs numpy | Val loss | Peak VRAM GB |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    base = {(r["kind"], r["tier"], r["batch"]): r for r in rows if r["backend"] == "numpy"}
    base_any = {(r["kind"], r["tier"]): r for r in rows if r["backend"] == "numpy"}
    for r in rows:
        b = base.get((r["kind"], r["tier"], r["batch"])) or base_any.get((r["kind"], r["tier"]))
        speed = f"{r['rows_per_second'] / b['rows_per_second']:.1f}x" if b and b["rows_per_second"] and r["rows_per_second"] else ""
        rate = r["rows_per_second"] if r["rows_per_second"] is not None else "too short"
        out.append(f"| {r['kind']} | {r['tier']} | {r['backend']} | {r['batch']} | {rate} | {r['epoch_seconds']} | "
                   f"{r['wall_seconds']} | {speed} | {r['val']:.4f} | {r['peak_reserved_gb'] or ''} |")
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--model", default="n15")
    ap.add_argument("--tier", nargs="+", default=["S", "XXL"], choices=list(TIERS))
    ap.add_argument("--backends", nargs="+", default=["numpy", "torch-cuda"], choices=["numpy", "torch-cpu", "torch-cuda"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1024])
    ap.add_argument("--kinds", nargs="+", default=["policy", "value"], choices=["policy", "value"])
    ap.add_argument("--max-rows", type=int, default=200_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--out", default=str(REPO / "reports" / "host"))
    a = ap.parse_args(argv)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for kind in a.kinds:
            for tier in a.tier:
                for backend in a.backends:
                    for batch in a.batches:
                        print(f"[bench] {kind} {tier} {backend} batch {batch}", flush=True)
                        rows.append(run_one(kind, tier, backend, batch, a, Path(tmp)))
    meta = {"host": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(),
            "model": a.model, "max_rows": a.max_rows, "epochs": a.epochs, "tf32": a.tf32,
            "gpu": next((r["gpu"] for r in rows if r["gpu"]), None), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    (out / f"training_bench_{host}.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    (out / f"training_bench_{host}.md").write_text(markdown(rows, meta), encoding="utf-8")
    print(markdown(rows, meta))


if __name__ == "__main__":
    main()
