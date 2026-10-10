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
        self.log_path.write_text("", encoding="utf-8")

    def log(self, **row):
        print("  ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def save(self, state: dict, **extra):
        np.savez(self.dir / "model.npz", **state)
        (self.dir / "config.json").write_text(json.dumps(self.config | extra, indent=1), encoding="utf-8")


def run_features(schema: dict, family: str) -> dict:
    """Config entries that bind a run to its feature family: `features`, plus the legacy `spec_version` for p2/v2."""
    from fusion import families  # noqa: PLC0415
    out = {"features": families.record(schema, family)}
    if family in families.LEGACY.values():
        out["spec_version"] = schema["spec_version"]
    return out


def add_training_args(ap, batch: int):
    """Arguments shared by train_policy and train_value."""
    ap.add_argument("--shards", default="data/shards")
    ap.add_argument("--model", required=True, help="model id from profiles/: n02 ... n15 or general")
    ap.add_argument("--out", required=True, help="run directory (convention: runs/<policy|value>_<model>)")
    ap.add_argument("--init", help="run directory to warm-start from (convention: the profile's init_from model)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--patience", type=int, default=2, help="epochs without improvement before a decay or a stop")
    ap.add_argument("--plateau-decay", type=int, default=1, help="times the learning rate halves on a plateau before stopping")
    ap.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=batch)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--max-rows", type=int, default=0, help="subsample the training split (0 = all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-split", default="val", help="split for early stopping and calibration")
    from fusion.capacity import TIERS  # noqa: PLC0415
    ap.add_argument("--capacity", choices=tuple(TIERS),
                    help="shared architecture preset; hidden-layer flags override individual heads")
    ap.add_argument("--max-width", type=int, default=0,
                    help="widest hidden layer allowed (default: selected preset, else 256; explicit custom wide layers need this)")
    ap.add_argument("--count-weighting", choices=("none", "inv_teams"), default="none",
                    help="scale row weights by 1 / teams so team counts with more teams do not dominate general models")
    ap.add_argument("--backend", choices=("numpy", "torch"), default="numpy",
                    help="numpy (reference) or torch (fusion/torch_backend.py: same objective, CUDA or CPU)")
    ap.add_argument("--device", default="auto", help="torch backend: auto, cuda, cuda:N or cpu")
    ap.add_argument("--tf32", action="store_true", help="torch backend: allow TF32 matmuls on CUDA (faster, other rounding)")
    ap.add_argument("--gpu-resident-gb", type=float, default=6.0,
                    help="torch backend: keep a split's rows on the device when they fit this budget, else stream them")
    ap.add_argument("--gpu-window-rows", type=int, default=1_000_000,
                    help="torch backend: rows per streamed window (rounded down to whole minibatches)")


def resolve_hidden(args, kind: str):
    """Resolve a P/V preset and CLI overrides before loading the expensive dataset.

    Existing custom-width calls keep their explicit-limit contract. Selecting a
    named preset supplies that preset's ceiling; a smaller explicit ceiling is
    still rejected. Actual dimensions are stored in every run as before.
    """
    from fusion.capacity import POLICY_HEADS, hidden_buffer_width, preset  # noqa: PLC0415
    from fusion.quant import MAX_WIDTH  # noqa: PLC0415
    if kind not in ("policy", "value"):
        raise ValueError("kind must be policy or value")
    base = preset(args.capacity or "S")
    heads = POLICY_HEADS if kind == "policy" else ("value",)
    selected = {h: base[h] for h in heads}
    if args.max_width < 0:
        raise SystemExit("--max-width must be positive, or 0 for the default ceiling")
    ceiling = args.max_width or (hidden_buffer_width(selected) if args.capacity else MAX_WIDTH)
    result, custom = {}, False
    for head in heads:
        flag = "hidden" if kind == "value" else f"hidden_{head}"
        override = getattr(args, flag)
        custom |= override is not None
        widths = override.split(",") if override is not None else selected[head]
        result[head] = check_widths(widths, "--" + flag.replace("_", "-"), ceiling)
    args.max_width = ceiling
    label = args.capacity or ("custom" if custom else "S")
    return (result if kind == "policy" else result["value"]), label


def policy_rows(d: dict, shards, keep_unlabeled: bool = False) -> np.ndarray:
    """Rows that are opponent-policy samples: a valid reply, from a family that gives labels."""
    from fusion.paths import is_unlabeled, unlabeled_families  # noqa: PLC0415
    from training.dataset import load_index  # noqa: PLC0415
    keep = d["valid"].copy()
    if not keep_unlabeled:
        names = load_index(shards)["families"]
        unlabeled = unlabeled_families()
        drop = [i for i, name in enumerate(names) if is_unlabeled(name, unlabeled)]
        keep &= ~np.isin(d["family"], drop)
    return keep


def data_manifest(shards, counts) -> dict:
    """What a run trained on: the dataset manifest restricted to its team counts, plus a fingerprint."""
    from training.dataset import data_digest, load_index  # noqa: PLC0415
    index = load_index(shards)
    man = index.get("manifest", {})
    keys = {f"n{t:02d}" for t in counts}
    return {"team_counts": list(counts), "digest": data_digest(index, counts),
            "matches_by_count": {k: v for k, v in man.get("matches_by_count", {}).items() if k in keys},
            "rows": {k: v for k, v in index["counts"].items() if k.split("/")[0] in keys},
            "registries": man.get("registries", []), "regimes": man.get("regimes", {}),
            "ai_test_commits": man.get("ai_test_commits"), "options": index["options"]}


def check_widths(widths, what: str, limit: int | None = None) -> list[int]:
    """Hidden widths must fit the C++ scratch buffers (inference/mlp.hpp kMaxWidth); fail before training, not at export."""
    from fusion.quant import MAX_WIDTH  # noqa: PLC0415
    limit = limit or MAX_WIDTH
    widths = [int(w) for w in widths]
    if any(w < 1 or w > limit for w in widths):
        raise SystemExit(f"{what} widths {widths}: each must be 1..{limit} (inference/mlp.hpp kMaxWidth; "
                         "pass --max-width for a bot built with -DFUSION_MAX_WIDTH)")
    return widths


def warm_start(model, init_run, schema: dict | None = None) -> str | None:
    """Initialize from another run of the same feature family (fusion/families.py) and the same shapes."""
    if not init_run:
        return None
    if schema is not None:
        from fusion import families  # noqa: PLC0415
        cfg = json.loads((Path(init_run) / "config.json").read_text(encoding="utf-8"))
        feats = families.run_features(cfg, families.kind(model.family))
        if feats["family"] != model.family:
            raise SystemExit(f"--init {init_run} was trained on feature family {feats['family']}, this run uses {model.family}")
        families.check(schema, feats, f"--init {init_run}")
    state = dict(np.load(Path(init_run) / "model.npz"))
    mine = model.state()
    for k, v in mine.items():
        if k in state and np.shape(state[k]) != np.shape(v):
            raise SystemExit(f"--init {init_run}: {k} has shape {np.shape(state[k])}, this model needs {np.shape(v)}")
    model.load({k: state.get(k, v) for k, v in mine.items()})
    return str(init_run)


def torch_fit(make_trainer, args, run: Run, rng):
    """fit() on the torch backend; the trained (best) weights end in the NumPy model the trainer wraps."""
    from fusion import torch_backend as tb  # noqa: PLC0415  (optional dependency)
    dev = tb.pick_device(args.device)
    env = tb.configure(dev, args.tf32)
    trainer = make_trainer(tb, dev)
    print(f"[torch] {env} train rows {'resident' if trainer.train.resident else 'streamed'}, "
          f"val rows {'resident' if trainer.val.resident else 'streamed'}", flush=True)
    best = fit(trainer, trainer.train_epoch, trainer.evaluate, args, run, rng, trainer.opt)
    run.config["backend"] = env | {"timing": trainer.timing} | tb.peak_memory(dev)
    return best


def fit(model, train_step, val_loss, args, run: Run, rng, opt):
    """Epoch loop with a learning-rate schedule, plateau decay and early stopping. Keeps the best state.

    On `patience` epochs without improvement the learning rate halves (up to
    --plateau-decay times) and training continues from the best state; after
    that it stops.
    """
    best, best_state, waited, decays = float("inf"), None, 0, 0
    scale = 1.0
    for epoch in range(1, args.epochs + 1):
        if args.lr_schedule == "cosine":
            lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + np.cos(np.pi * (epoch - 1) / max(args.epochs - 1, 1)))
        else:
            lr = args.lr
        opt.lr = lr * scale
        t0 = time.time()
        train = train_step(rng)
        val = val_loss()
        if not (np.isfinite(train) and np.isfinite(val)):
            raise FloatingPointError(f"non-finite loss at epoch {epoch}: train {train}, val {val}")
        run.log(epoch=epoch, lr=float(opt.lr), train=train, val=val, seconds=round(time.time() - t0, 3))
        if val < best - 1e-5:
            best, best_state, waited = val, {k: v.copy() for k, v in model.state().items()}, 0
            continue
        waited += 1
        if waited < args.patience:
            continue
        if decays >= args.plateau_decay:
            break
        decays, waited, scale = decays + 1, 0, scale * 0.5
        model.load(best_state)
    model.load(best_state)
    return best
