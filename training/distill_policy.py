"""Bounded-memory distillation and QAT entry point for the opponent policy P (task T0030; no real run in that task).

    python -m training.distill_policy teacher-profile                       # 6XL dimensions and memory estimates
    python -m training.distill_policy teacher-targets --teacher RUN --cache DIR --split train --out build/p_distill/t_train
    python -m training.distill_policy train --stage warmup  --cache DIR --student xxl --out runs/s_warm
    python -m training.distill_policy train --stage distill --cache DIR --teacher-shard build/p_distill/t_train \
        --init runs/s_warm --alpha 0.5 --tau 1.0 --out runs/s_distill
    python -m training.distill_policy train --stage qat --profile mixed4_8 --backend torch --cache DIR \
        --teacher-shard build/p_distill/t_train --init runs/s_distill --out runs/s_qat
    python -m training.distill_policy toy --out build/p_distill/toy          # every stage on synthetic rows (tests)

Row caches (--cache) are directories of NumPy arrays opened memory-mapped: rec.npy (R, record width) int16, y_move,
y_joint, y_act, y_tgt, weight, and row.npy (stable row ids). T0011's data/pf_policy_cache/<split> has no row.npy:
its receipt CACHE.json (row count, shapes, dtypes, data SHA-256 per array) stands in, the row ids are the
positions in that fixed order, and the cache identity is derived from the receipt's digests. Training reads one minibatch of rows at a time, in a
seeded permutation, so memory is bounded by the batch (a later data task writes the cache, as T0011's
data/pf_policy_cache: never a concatenated copy in RAM). Teacher shards (fusion/policy_distill.py) are written
batch by batch and checked against every student batch: rows, masks and movement contexts must be identical.

Stages: warmup = supervised CE (the trainer's loss, alpha = 1); distill = alpha CE + (1 - alpha) tau^2 KL;
--backend torch runs warmup and distill on the float student on a device (fusion.torch_qat.FloatPolicy, batches
built on the device from the int16 records); qat always runs there. qat = the same loss through fusion/torch_qat.py, whose forward values are the deployed integer outputs of the
chosen low-bit profile. Every stage writes config.json (identities of data, teacher shard, init, profile, loss)
and a receipt (steps, rows, final losses) next to model.npz, as training/common.Run does.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from fusion import policy_distill as D
from fusion import precision
from fusion.models import PolicyNet, policy_batch
from fusion.nn import Adam
from fusion.schema import layout as feature_layout

LABELS = ("y_move", "y_joint", "y_act", "y_tgt")
CACHE_FIELDS = ("rec", *LABELS, "weight", "row")
STUDENTS = {"xxl": D.STUDENT_XXL, "6xl": D.TEACHER_6XL,
            "s": {"mover": [16], "squad": [64, 32], "act": [16], "target": [8]}}


class RowCache:
    """Memory-mapped training rows; slices are copied per batch only."""

    def __init__(self, path: Path):
        self.path = Path(path)
        receipt = self.path / "CACHE.json"
        self.receipt = None
        if not (self.path / "row.npy").is_file() and receipt.is_file():
            self.receipt = json.loads(receipt.read_text(encoding="utf-8"))   # T0011's layout: positions are row ids
        fields = CACHE_FIELDS if self.receipt is None else tuple(k for k in CACHE_FIELDS if k != "row")
        self.arr = {k: np.load(self.path / f"{k}.npy", mmap_mode="r") for k in fields}
        if (self.path / "teams.npy").is_file():           # optional: team count per row (--count-weighting inv_teams)
            self.arr["teams"] = np.load(self.path / "teams.npy", mmap_mode="r")
        n = {k: len(v) for k, v in self.arr.items()}
        if len(set(n.values())) != 1:
            raise ValueError(f"{path}: fields of different lengths {n}")
        self.rows = n["rec"]
        if self.receipt is not None:
            r = self.receipt
            if not r.get("complete") or r.get("rows") != self.rows:
                raise ValueError(f"{receipt}: incomplete, or {r.get('rows')} rows where the arrays hold {self.rows}")
            for k, a in self.arr.items():
                m = r["arrays"].get(k)
                if m is None or list(a.shape) != m["shape"] or a.dtype.str != m["dtype"]:
                    raise ValueError(f"{receipt}: {k} is {a.dtype.str}{list(a.shape)}, the receipt says {m}")

    def close(self) -> None:
        """Release the mapped files (Windows cannot delete a file that is still mapped)."""
        for v in self.arr.values():
            mm = getattr(v, "_mmap", None)
            if mm is not None:
                mm.close()
        self.arr = {}

    def identity(self) -> str:
        h = hashlib.sha256(f"{self.rows}|{self.arr['rec'].shape}".encode())
        if self.receipt is None:
            h.update(np.ascontiguousarray(self.arr["row"]).tobytes())
        else:                   # the receipt's digests (not rehashed here: verify_receipt does that, once)
            h.update(json.dumps({"identity": self.receipt.get("identity"),
                                 "arrays": {k: self.receipt["arrays"][k]["data_sha256"] for k in sorted(self.arr)}},
                                sort_keys=True).encode())
        return h.hexdigest()[:16]

    def verify_receipt(self, chunk_bytes: int = 64 << 20) -> None:
        """Rehash every array against CACHE.json (sha256 of the C-order data, as T0011 wrote it), chunk by chunk."""
        if self.receipt is None:
            raise ValueError(f"{self.path}: no CACHE.json receipt to verify")
        for k, a in self.arr.items():
            h, step = hashlib.sha256(), max(1, chunk_bytes // max(1, a[:1].nbytes))
            for i in range(0, len(a), step):
                h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
            if h.hexdigest() != self.receipt["arrays"][k]["data_sha256"]:
                raise ValueError(f"{self.path / (k + '.npy')}: data SHA-256 differs from CACHE.json")

    def take(self, idx: np.ndarray) -> dict:
        d = {k: np.asarray(self.arr[k][idx]) for k in self.arr}
        if self.receipt is not None:
            d["row"] = np.asarray(idx, np.int64)
        return d


def synthetic_rows(layout, n: int, seed: int = 0) -> dict:
    """Extractor-shaped rows inside the feature spec with consistent labels (contract tests and the toy run)."""
    rng = np.random.default_rng(seed)
    rec = np.zeros((n, layout.width), np.int16)

    def put(name, values):
        off, shape = layout.blocks[name]
        rec[:, off:off + int(np.prod(shape))] = values.reshape(n, -1)

    for name, group in (("g", "global"), ("u", "unit"), ("m", "move"), ("cand_f", "cand_f"), ("pc", "combat")):
        _, shape = layout.blocks[name]
        lo, hi = layout.bounds(group)
        put(name, rng.integers(lo, hi + 1, (n, *shape)))
    legal = rng.random((n, 3, 5)) < 0.6
    legal[:, :, 0] = True
    put("legal", legal.astype(np.int16))
    act = rng.random((n, 3, 5, 4)) < 0.5
    act[..., 0] = True
    put("act_legal", act.astype(np.int16))
    cand = np.where(rng.random((n, 3, 8)) < 0.6, rng.integers(0, 30, (n, 3, 8)), -1)
    put("cand", cand)
    move = np.array([[rng.choice(np.nonzero(legal[b, k])[0]) for k in range(3)] for b in range(n)])
    y_act = np.array([[rng.choice(np.nonzero(act[b, k, move[b, k]])[0]) for k in range(3)] for b in range(n)])
    valid = cand >= 0
    y_tgt = np.array([[rng.choice(np.nonzero(valid[b, k])[0]) if valid[b, k].any() and y_act[b, k] == 3 else -1
                       for k in range(3)] for b in range(n)])
    return {"rec": rec, "y_move": move.astype(np.int8), "y_joint": (move[:, 0] * 25 + move[:, 1] * 5 + move[:, 2]).astype(np.int16),
            "y_act": y_act.astype(np.int8), "y_tgt": y_tgt.astype(np.int8), "weight": rng.uniform(0.5, 2.0, n).astype(np.float32)}


def write_cache(path: Path, d: dict, rows: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for k in CACHE_FIELDS:
        np.save(path / f"{k}.npy", rows if k == "row" else d[k])


def teacher_targets(teacher: PolicyNet, layout, cache: RowCache, split: str, out: Path, tau: float, batch: int,
                    dataset: str = "") -> dict:
    """Teacher distributions for every cache row, in cache order, `batch` rows at a time."""
    from fusion.precision_calib import check_split  # noqa: PLC0415  (a test split never feeds training targets)

    check_split(split)
    C = layout.blocks["cand"][1][-1]
    w = D.TeacherShardWriter(out, cache.rows, C, {"teacher": D.state_digest(teacher), "family": teacher.family,
                                                  "split": split, "tau": tau, "teacher_beta": dict(teacher.beta),
                                                  "dataset": dataset or cache.identity()})
    for a in range(0, cache.rows, batch):
        idx = np.arange(a, min(cache.rows, a + batch))
        d = cache.take(idx)
        X = policy_batch(d["rec"], layout, True, teacher.family)
        mv = d["y_move"].astype(np.int64)
        w.write(d["row"], X, mv, D.teacher_probs(teacher, X, mv, tau))
    return w.close()


def row_weights(d: dict, args) -> np.ndarray:
    """The rows' weights, divided by the team count under --count-weighting inv_teams (as training/train_policy.py)."""
    w = d["weight"]
    if getattr(args, "count_weighting", "none") == "inv_teams":
        if "teams" not in d:
            raise SystemExit("--count-weighting inv_teams needs teams.npy in the cache")
        w = (w / d["teams"].astype(np.float32)).astype(np.float32)
    return w


class _Model:
    """state()/load() for training.common.fit: the NumPy student, synchronized with its device copy."""

    def __init__(self, student: PolicyNet, dev_policy=None):
        self.student, self.dev_policy = student, dev_policy

    def state(self):
        if self.dev_policy is not None:
            self.dev_policy.to_ref()
        return self.student.state()

    def load(self, st):
        self.student.load(st)
        if self.dev_policy is not None:
            self.dev_policy.from_ref()


def train(stage: str, student: PolicyNet, layout, cache: RowCache, args, shard: D.TeacherShard | None, out: Path,
          val: RowCache | None = None) -> dict:
    """One stage. Without `val`: args.epochs passes at a constant learning rate (contract tests). With `val`: the
    epoch loop of training.common.fit as T0011 trained P (cosine schedule from --lr to --min-lr, the supervised
    validation loss each epoch, best state kept, --patience / --plateau-decay), then, for warmup and distill,
    temperature calibration on the validation rows (training/train_policy.calibrate). qat is not calibrated here:
    the export fits its temperatures on the deployed outputs."""
    cfg = D.DistillConfig(alpha=1.0 if stage == "warmup" else args.alpha, tau=args.tau, act_weight=args.act_weight,
                          target_weight=args.target_weight, smoothing=args.smoothing)
    cfg.check()
    if stage != "warmup" and shard is None:
        raise SystemExit(f"stage {stage} needs --teacher-shard")
    if shard is not None and shard.identity["family"] != student.family:
        raise SystemExit(f"teacher shard is family {shard.identity['family']}, student {student.family}")
    if shard is not None and (shard.identity.get("dataset") != cache.identity() or shard.identity.get("rows") != cache.rows):
        raise SystemExit(f"teacher shard was written for cache {shard.identity.get('dataset')} "
                         f"({shard.identity.get('rows')} rows), not this cache {cache.identity()} ({cache.rows} rows)")
    backend = "torch" if stage == "qat" else getattr(args, "backend", "numpy")
    rng = np.random.default_rng(args.seed)
    steps, last = 0, {}
    torch_state, qat = None, None
    if backend == "torch":
        import torch  # noqa: PLC0415

        from fusion import torch_qat  # noqa: PLC0415
        from fusion.torch_backend import DeviceLayout  # noqa: PLC0415
        dev = torch.device(args.device)
        qat = torch_qat.QATPolicy(student, layout, args.profile, dev) if stage == "qat" else torch_qat.FloatPolicy(student, dev)
        dl = None if stage == "qat" else DeviceLayout(layout, dev)
        opt = torch.optim.SGD(qat.params, lr=args.lr) if args.optimizer == "sgd" else None
        if opt is None:
            from fusion.torch_backend import TorchAdam  # noqa: PLC0415
            opt = TorchAdam(qat.params, lr=args.lr, weight_decay=args.weight_decay)
        torch_state = (torch, torch_qat, qat, opt, dev, dl)
    else:
        opt = Adam(student.params, lr=args.lr, weight_decay=args.weight_decay)
    def epoch(r) -> float:
        nonlocal steps, last
        order = r.permutation(cache.rows)
        total, wsum = 0.0, 0.0
        for a in range(0, cache.rows, args.batch):
            idx = np.sort(order[a:a + args.batch])     # sorted: contiguous mmap reads, and shard slices by position
            d = cache.take(idx)
            d["weight"] = row_weights(d, args)
            mv = d["y_move"].astype(np.int64)
            teacher = None
            if shard is not None:
                Xn = policy_batch(d["rec"], layout, True, student.family)
                teacher = _shard_rows(shard, idx, d["row"], Xn, mv)
            if torch_state is None:
                X = policy_batch(d["rec"], layout, True, student.family)
                student.zero_grad()
                last = D.distill_loss(student, X, d, d["weight"], teacher, cfg, backward=True)
                opt.step(student.grads)
            else:
                torch, torch_qat, qat, topt, dev, dl = torch_state
                if dl is None:      # qat: raw integer inputs, as deployed
                    T = torch_qat.tensors(policy_batch(d["rec"], layout, False, student.family), dev)
                else:               # float student: normalized features built on the device
                    T = torch_qat.device_batch(torch.from_numpy(np.ascontiguousarray(d["rec"])).to(dev), dl, student.family)
                y = {k: torch.tensor(d[k].astype(np.int64), device=dev) for k in LABELS}
                tt = None if teacher is None else {k: torch.tensor(v, device=dev) for k, v in teacher.items()}
                topt.zero_grad()
                loss, parts = torch_qat.qat_loss(qat, T, y, torch.tensor(d["weight"], device=dev), tt, cfg)
                loss.backward()
                topt.step()
                last = {k: float(v) for k, v in parts.items()}
            steps += 1
            total += float(last["total"]) * float(d["weight"].sum())
            wsum += float(d["weight"].sum())
        return total / max(wsum, 1e-12)

    out.mkdir(parents=True, exist_ok=True)
    best_val, calibrated = None, False
    if val is None:
        for _ in range(args.epochs):
            epoch(rng)
        if torch_state is not None:
            qat.to_ref()
    else:
        from training.common import Run, fit  # noqa: PLC0415
        from training.train_policy import calibrate  # noqa: PLC0415
        if not hasattr(opt, "t"):
            raise SystemExit("the scheduled epoch loop (--val-cache) needs --optimizer adam")
        sup = D.DistillConfig(alpha=1.0, tau=cfg.tau, act_weight=cfg.act_weight, target_weight=cfg.target_weight,
                              smoothing=cfg.smoothing)
        run = Run(out, {})
        model = _Model(student, qat)
        best_val = fit(model, epoch, lambda: val_loss(student, layout, val, args, sup, qat), args, run, rng, opt,
                       on_best=lambda e, v, st: np.savez(out / "best.npz", **st))
        if stage != "qat":
            vd = {k: val.arr[k] for k in ("rec", *LABELS)} | {"weight": row_weights({k: np.asarray(val.arr[k]) for k in
                                                                                  ("weight", "teams") if k in val.arr}, args)}
            calibrate(student, layout, vd)
            calibrated = True
    np.savez(out / "model.npz", **student.state())
    config = {"kind": "policy-distill", "stage": stage, "family": student.family, "hidden": student.hidden,
              "loss": {"alpha": cfg.alpha, "tau": cfg.tau, "act_weight": cfg.act_weight,
                       "target_weight": cfg.target_weight, "smoothing": cfg.smoothing},
              "cache": cache.identity(), "rows": cache.rows, "teacher_shard": None if shard is None else shard.identity,
              "backend": backend, "device": str(args.device) if backend == "torch" else "cpu",
              "profile": args.profile if stage == "qat" else None,
              "profile_identity": precision.profile_identity(args.profile, precision.PROFILES[args.profile]) if stage == "qat" else None,
              "init": args.init, "seed": args.seed, "epochs": args.epochs, "batch": args.batch,
              "count_weighting": getattr(args, "count_weighting", "none"),
              "schedule": None if val is None else {k: getattr(args, k) for k in
                                                    ("lr", "min_lr", "lr_schedule", "patience", "plateau_decay", "weight_decay")},
              "val_cache": None if val is None else val.identity(), "val_rows": None if val is None else val.rows}
    receipt = {"steps": steps, "final": {k: float(v) for k, v in last.items()}, "model_digest": D.state_digest(student),
               "best_val": best_val, "calibrated": calibrated, "beta": {k: float(v) for k, v in student.beta.items()}}
    (out / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True, default=str), encoding="utf-8")
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, sort_keys=True), encoding="utf-8")
    return {"config": config, "receipt": receipt}


def val_loss(student: PolicyNet, layout, val: RowCache, args, sup: D.DistillConfig, dev_policy=None, size: int = 8192) -> float:
    """The supervised loss (alpha = 1, the trainer's validation objective) over every validation row, weighted;
    on the device policy (float or QAT forward) when there is one."""
    total, wsum = 0.0, 0.0
    for a in range(0, val.rows, size):
        d = val.take(np.arange(a, min(val.rows, a + size)))
        w = row_weights(d, args)
        if dev_policy is None:
            out = D.distill_loss(student, policy_batch(d["rec"], layout, True, student.family), d, w, None, sup, False)
            v = float(out["total"])
        else:
            import torch  # noqa: PLC0415

            from fusion import torch_qat  # noqa: PLC0415
            dev = torch.device(args.device)
            with torch.no_grad():
                if isinstance(dev_policy, torch_qat.QATPolicy):
                    T = torch_qat.tensors(policy_batch(d["rec"], layout, False, student.family), dev)
                else:
                    from fusion.torch_backend import DeviceLayout  # noqa: PLC0415
                    T = torch_qat.device_batch(torch.from_numpy(np.ascontiguousarray(d["rec"])).to(dev),
                                               DeviceLayout(layout, dev), student.family)
                y = {k: torch.tensor(d[k].astype(np.int64), device=dev) for k in LABELS}
                _, parts = torch_qat.qat_loss(dev_policy, T, y, torch.tensor(w, device=dev), None, sup)
            v = float(parts["total"])
        total += v * float(w.sum())
        wsum += float(w.sum())
    return total / max(wsum, 1e-12)


def _shard_rows(shard: D.TeacherShard, idx: np.ndarray, rows: np.ndarray, X, mv) -> dict:
    """Teacher targets of an arbitrary (sorted) set of cache positions, checked like TeacherShard.batch."""
    if not np.array_equal(shard.arr["row"][idx], rows):
        raise ValueError("row-identity mismatch: the teacher shard's rows are not the student batch's rows")
    if not np.array_equal(shard.arr["move"][idx], mv):
        raise ValueError("movement-context mismatch: teacher and student must use identical moves")
    jm, am, tm = D.masks(X, mv)
    for k, m in (("joint_mask", jm), ("act_mask", am), ("target_mask", tm)):
        if not np.array_equal(shard.arr[k][idx], m):
            raise ValueError(f"{k} mismatch between the teacher shard and the student batch")
    return {k: np.asarray(shard.arr[k][idx]) for k in ("joint", "act", "target")}


def load_student(layout, spec: str, family: str, init: Path | None, seed: int) -> PolicyNet:
    hidden = STUDENTS[spec] if spec in STUDENTS else json.loads(spec)
    m = PolicyNet(layout, np.random.default_rng(seed), hidden, family)
    if init:
        m.load(dict(np.load(Path(init) / "model.npz")))
    return m


def add_loss_args(p):
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--act-weight", type=float, default=0.5)
    p.add_argument("--target-weight", type=float, default=0.5)
    p.add_argument("--smoothing", type=float, default=0.01)


def toy(out: Path, with_torch: bool) -> dict:
    """Every stage end to end on 96 synthetic rows (CPU): the contract test of this entry point, not training."""
    layout = feature_layout()
    d = synthetic_rows(layout, 96, seed=21)
    cache_dir, val_dir = out / "cache", out / "val"
    write_cache(cache_dir, d, np.arange(1000, 1096))
    cache = RowCache(cache_dir)
    teacher = PolicyNet(layout, np.random.default_rng(2), {"mover": [8], "squad": [32, 16], "act": [8], "target": [4]})
    shard_ident = teacher_targets(teacher, layout, cache, "train", out / "teacher", tau=2.0, batch=32)
    shard = D.TeacherShard(out / "teacher", {"family": "p2", "split": "train"})
    args = argparse.Namespace(alpha=0.5, tau=2.0, act_weight=0.5, target_weight=0.5, smoothing=0.01, seed=0,
                              epochs=1, batch=32, lr=1e-3, weight_decay=0.0, init=None, profile="mixed4_8",
                              device="cpu", optimizer="adam")
    res = {"teacher": shard_ident}
    student = load_student(layout, "s", "p2", None, 0)
    try:
        res["warmup"] = train("warmup", student, layout, cache, args, None, out / "warmup")
        res["distill"] = train("distill", student, layout, cache, args, shard, out / "distill")
        if with_torch:
            res["qat"] = train("qat", student, layout, cache, args, shard, out / "qat")
    finally:
        shard.close()
        cache.close()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("teacher-profile").add_argument("--family", choices=("p2", "p3"), default="p2")
    t = sub.add_parser("teacher-targets")
    t.add_argument("--teacher", type=Path, required=True)
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--split", required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--tau", type=float, default=1.0)
    t.add_argument("--batch", type=int, default=8192)
    r = sub.add_parser("train")
    r.add_argument("--stage", choices=("warmup", "distill", "qat"), required=True)
    r.add_argument("--cache", type=Path, required=True)
    r.add_argument("--teacher-shard", type=Path)
    r.add_argument("--student", default="xxl", help="xxl, 6xl, s, or a JSON hidden-width dict")
    r.add_argument("--family", choices=("p2", "p3"), default="p2")
    r.add_argument("--init", type=Path)
    r.add_argument("--profile", choices=("uniform8", "mixed4_8"), default="mixed4_8")
    r.add_argument("--backend", choices=("numpy", "torch"), default="numpy")
    r.add_argument("--device", default="cpu")
    r.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    r.add_argument("--epochs", type=int, default=1)
    r.add_argument("--batch", type=int, default=1024)
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--weight-decay", type=float, default=1e-5)
    r.add_argument("--val-cache", type=Path, help="validation rows: enables the scheduled epoch loop and calibration")
    r.add_argument("--lr-schedule", choices=("constant", "cosine"), default="cosine")
    r.add_argument("--min-lr", type=float, default=1e-4)
    r.add_argument("--patience", type=int, default=2)
    r.add_argument("--plateau-decay", type=int, default=1)
    r.add_argument("--count-weighting", choices=("none", "inv_teams"), default="none")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--out", type=Path, required=True)
    add_loss_args(r)
    y = sub.add_parser("toy")
    y.add_argument("--out", type=Path)
    y.add_argument("--torch", action="store_true")
    v = sub.add_parser("verify-cache", help="rehash a T0011-format cache against its CACHE.json")
    v.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args(argv)
    layout = feature_layout()
    if a.cmd == "teacher-profile":
        print(json.dumps({"6xl": D.teacher_estimates(layout, a.family), "xxl": D.teacher_estimates(layout, a.family, D.STUDENT_XXL)},
                         indent=1))
        return 0
    if a.cmd == "teacher-targets":
        from training.export_weights import load_policy  # noqa: PLC0415
        teacher = load_policy(a.teacher, layout)
        print(json.dumps(teacher_targets(teacher, layout, RowCache(a.cache), a.split, a.out, a.tau, a.batch), indent=1))
        return 0
    if a.cmd == "verify-cache":
        c = RowCache(a.cache)
        c.verify_receipt()
        print(json.dumps({"cache": str(a.cache), "rows": c.rows, "identity": c.identity(), "verified": True}))
        return 0
    if a.cmd == "toy":
        if a.out:
            print(json.dumps(toy(a.out, a.torch), indent=1, default=str))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                print(json.dumps(toy(Path(tmp), a.torch), indent=1, default=str))
        return 0
    if a.stage == "qat" and a.backend != "torch":
        raise SystemExit("--stage qat runs on the Torch backend (fusion/torch_qat.py): add --backend torch")
    student = load_student(layout, a.student, a.family, a.init, a.seed)
    shard = D.TeacherShard(a.teacher_shard, {"family": a.family}) if a.teacher_shard else None
    res = train(a.stage, student, layout, RowCache(a.cache), a, shard, a.out, RowCache(a.val_cache) if a.val_cache else None)
    print(json.dumps(res["receipt"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
