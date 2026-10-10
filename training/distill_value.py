"""Distil a value network into a cheaper student (deployment Route B).

    python -m training.distill_value --teacher runs/v_pilot_D_s1 --cache data/vcache/v12xxl_v3 --soft-weight 0.5 \\
        --hidden 512,64 --rank data/rank/v12xxl --rank-weight 0.5 --backend torch --device cuda --checkpoint \\
        --out runs/v_pilot_E_s1

The student learns from three signals on the teacher's own training rows:
  - real outcomes (the primary target; validation, early stopping and calibration use them only);
  - the teacher's calibrated probability, mixed into the outcome target with --soft-weight
    ((1 - w) x real + w x teacher; BCE is linear in its target, so this is the weighted sum of both losses);
  - optionally, candidate-pair ranking on real continuation outcomes (--rank, --rank-weight; training.train_value).

A student cannot reproduce distinctions its inputs omit, so it must read the teacher's feature family (v3 for
a v3 teacher); a different family is refused. Any hidden widths are allowed, e.g. a small M-sized network or a
wide first layer with a small tail (4096,16), whose cost a first-layer accumulator mostly removes.

Writes <out>/teacher_train.npy (+ .json: teacher run and checkpoint hash, cache, rows) and then trains the
student through training.train_value with --soft-targets; the run's config records the teacher.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from fusion.paths import sha256
from fusion.schema import layout as feature_layout
from training import train_value
from training.export_weights import load_value
from training.value_cache import cache_identity, load_cache


def teacher_probs(teacher, beta: float, v: np.ndarray, layout, device: str, size: int = 32768) -> np.ndarray:
    scale = layout.value_scale(teacher.family)
    out = np.empty(len(v), np.float32)
    if device != "numpy":
        import torch  # noqa: PLC0415

        from fusion import torch_backend as tb  # noqa: PLC0415
        dev = tb.pick_device(device)
        tb.configure(dev, False)  # teacher targets use the same FP32 reference policy
        net = tb.TorchMLP(teacher.net, dev)
        sc = torch.tensor(scale, dtype=torch.float64, device=dev)
        with torch.no_grad():
            for s in range(0, len(v), size):
                x = (torch.from_numpy(np.ascontiguousarray(v[s:s + size])).to(dev).double() / sc).float()
                z = net.forward(x)[:, 0].double()
                out[s:s + size] = torch.sigmoid(beta * z).cpu().numpy()
        return out
    for s in range(0, len(v), size):
        z = teacher.logits((np.asarray(v[s:s + size]) / scale).astype(np.float32))[:, 0].astype(np.float64)
        out[s:s + size] = 1 / (1 + np.exp(-beta * z))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", required=True, help="value run of the teacher")
    ap.add_argument("--cache", required=True, help="V-only cache (the student trains on its train split)")
    ap.add_argument("--soft-weight", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher-batch", type=int, default=2048, help="bounded teacher inference batch (5XL needs measured VRAM)")
    a, rest = ap.parse_known_args(argv)
    if a.teacher_batch < 1:
        raise SystemExit("--teacher-batch must be positive")
    started = time.perf_counter()
    L = feature_layout()
    teacher, beta = load_value(Path(a.teacher), L)
    family = teacher.family
    if "--value-family" in rest and rest[rest.index("--value-family") + 1] != family:
        raise SystemExit(f"the student must read the teacher's feature family ({family}): it cannot reproduce what its inputs omit")
    device = rest[rest.index("--device") + 1] if "--device" in rest and "torch" in rest else "numpy"
    train = load_cache(a.cache, "train", family)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    soft = teacher_probs(teacher, beta, train["v"], L, device, a.teacher_batch)
    np.save(out / "teacher_train.npy", soft)
    meta = {"teacher": str(a.teacher), "teacher_sha256": sha256(Path(a.teacher) / "model.npz"), "teacher_beta": beta,
            "family": family, "cache": str(a.cache), "rows": int(len(soft)),
            "cache_sha256": cache_identity(a.cache)["sha256"], "targets_sha256": sha256(out / "teacher_train.npy"),
            "teacher_batch": a.teacher_batch, "target_seconds": round(time.perf_counter() - started, 3)}
    (out / "teacher_train.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    model = next((rest[i + 1] for i, x in enumerate(rest) if x == "--model"), None)
    args = (["--model", "general"] if model is None else []) + rest + [
        "--cache", a.cache, "--value-family", family, "--soft-targets", str(out / "teacher_train.npy"),
        "--soft-weight", str(a.soft_weight), "--out", a.out]
    print(f"[distill] teacher {a.teacher} ({family}), {len(soft)} rows, soft weight {a.soft_weight}", flush=True)
    train_value.main(args)
    (out / "distill_pipeline.json").write_text(json.dumps({"target_seconds": meta["target_seconds"],
        "pipeline_seconds": round(time.perf_counter() - started, 3)}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
