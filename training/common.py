"""Shared training utilities: minibatches, early stopping, run directories."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from fusion.paths import REPO


def batches(n: int, size: int, rng: np.random.Generator | None):
    order = rng.permutation(n) if rng is not None else np.arange(n)
    for start in range(0, n, size):
        yield order[start:start + size]


def subset(arrays: dict, idx) -> dict:
    return {k: v[idx] for k, v in arrays.items()}


def repo_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class Run:
    """A training run directory: config.json, log.jsonl, model.npz."""

    def __init__(self, out: str | Path, config: dict):
        self.dir = Path(out)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.config = config | {"repo_commit": repo_commit(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self.log_path = self.dir / "log.jsonl"
        self.log_path.write_text("")

    def log(self, **row):
        print("  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def save(self, state: dict, **extra):
        np.savez(self.dir / "model.npz", **state)
        (self.dir / "config.json").write_text(json.dumps(self.config | extra, indent=1))


def fit(model, train_step, val_loss, epochs: int, patience: int, run: Run, rng):
    """Generic loop: train_step(epoch) runs one epoch, val_loss() -> float. Keeps the best state."""
    best, best_state, waited = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train = train_step(rng)
        val = val_loss()
        if not (np.isfinite(train) and np.isfinite(val)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}: train {train}, val {val}")
        run.log(epoch=epoch, train=train, val=val, seconds=round(time.time() - t0, 1))
        if val < best - 1e-5:
            best, best_state, waited = val, {k: v.copy() for k, v in model.state().items()}, 0
        else:
            waited += 1
            if waited >= patience:
                break
    model.load(best_state)
    return best
