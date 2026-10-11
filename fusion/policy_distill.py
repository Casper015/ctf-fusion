"""Teacher/student distillation for the opponent policy P, NumPy core (task T0030, docs/P_MIXED_PRECISION.md).

Loss per head, with the existing head weights (act_weight, target_weight) and sample weights w:

    L = alpha * CE(student, recorded label) + (1 - alpha) * tau^2 * KL(teacher_tau || softmax(student / tau))

    joint   the masked 125-way joint movement distribution (never argmax, never independent marginals)
    act     P(wait, pickup, drop, attack | unit, move) at the recorded move of each unit
    target  P(candidate | unit attacks after move) at the same recorded move
            (teacher and student see identical movement contexts and legal masks)

Temperature semantics: the teacher's probabilities come from its calibrated logits (its fitted beta per head,
PolicyNet.calibrated) divided by tau; the student's raw (uncalibrated) logits are divided by tau inside the KL,
and its own temperature is fitted after training, on validation rows, as for any model. alpha = 0.5 and tau are
starting points to validate later, not tuned values.

Teacher outputs are produced and consumed in bounded batches and stored in memory-mapped shards
(TeacherShardWriter / TeacherShard) keyed by teacher checkpoint, dataset, split, feature family, row ids, masks
and movement contexts. A student batch whose rows, masks or contexts differ from the shard's is refused.

The 6XL teacher is a profile for a later training task (teacher_6xl): exact head dimensions and estimates of
parameters, MACs, activations and optimizer memory. It is never exported to a bot.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import JOINT, PolicyBatch, PolicyNet, policy_widths
from .nn import masked_log_softmax, softmax_xent

FORMAT = "teacher-shard-1"

# A teacher-only profile (never deployed). XXL is mover 256, squad 1024/256, act 256, target 32.
TEACHER_6XL = {"mover": [1024], "squad": [4096, 1024], "act": [1024], "target": [128]}
STUDENT_XXL = {"mover": [256], "squad": [1024, 256], "act": [256], "target": [32]}


@dataclass
class DistillConfig:
    alpha: float = 0.5               # weight of the real-label cross-entropy; 1 - alpha on the teacher KL
    tau: float = 1.0                 # distillation temperature (teacher and student divided by tau in the KL)
    act_weight: float = 1.0
    target_weight: float = 1.0
    smoothing: float = 0.0           # label smoothing of the CE term (the trainer's semantics)
    stages: list = field(default_factory=lambda: ["warmup", "distill", "qat"])

    def check(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.tau <= 0:
            raise ValueError("tau must be positive")


def _digest(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(f"{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def state_digest(model: PolicyNet) -> str:
    st = model.state()
    return _digest(*[st[k] for k in sorted(st)])


def masks(X: PolicyBatch, move: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return X.joint_mask, X.act_mask(move), X.cand_valid


def teacher_probs(teacher: PolicyNet, X: PolicyBatch, move: np.ndarray, tau: float) -> dict[str, np.ndarray]:
    """Teacher distributions at temperature tau from its calibrated logits; masked entries are exactly 0."""
    was = teacher.calibrated
    teacher.calibrated = True
    try:
        zj = teacher.joint_logits(X)
        za = teacher.act_logits(X, move)
        zt = teacher.target_logits(X, move)
    finally:
        teacher.calibrated = was
    jm, am, tm = masks(X, move)
    return {"joint": np.exp(masked_log_softmax(zj / tau, jm)).astype(np.float32),
            "act": np.exp(masked_log_softmax(za / tau, am)).astype(np.float32),
            "target": np.exp(masked_log_softmax(zt / tau, tm)).astype(np.float32)}


# ------------------------------------------------------------------------------------------ teacher shards


class TeacherShardWriter:
    """Preallocated memory-mapped arrays for N rows, filled batch by batch (memory bounded by one batch)."""

    def __init__(self, out: Path, rows: int, cand: int, identity: dict):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        mm = lambda name, shape, dt: np.lib.format.open_memmap(self.out / f"{name}.npy", "w+", dt, shape)  # noqa: E731
        self.arr = {"row": mm("row", (rows,), np.int64), "joint": mm("joint", (rows, JOINT), np.float32),
                    "act": mm("act", (rows, 3, 4), np.float32), "target": mm("target", (rows, 3, cand), np.float32),
                    "joint_mask": mm("joint_mask", (rows, JOINT), np.bool_), "act_mask": mm("act_mask", (rows, 3, 4), np.bool_),
                    "target_mask": mm("target_mask", (rows, 3, cand), np.bool_), "move": mm("move", (rows, 3), np.int8)}
        self.identity = dict(identity) | {"format": FORMAT, "rows": rows}
        self.filled = 0

    def write(self, row_ids: np.ndarray, X: PolicyBatch, move: np.ndarray, probs: dict):
        a, n = self.filled, len(row_ids)
        jm, am, tm = masks(X, move)
        for k, v in (("row", row_ids), ("joint", probs["joint"]), ("act", probs["act"]), ("target", probs["target"]),
                     ("joint_mask", jm), ("act_mask", am), ("target_mask", tm), ("move", move)):
            self.arr[k][a:a + n] = v
        self.filled += n

    def close(self) -> dict:
        if self.filled != self.identity["rows"]:
            raise ValueError(f"shard incomplete: {self.filled} of {self.identity['rows']} rows")
        for v in self.arr.values():
            v.flush()
        ident = self.identity | {"row_digest": _digest(np.asarray(self.arr["row"])),
                                 "mask_digest": _digest(*(np.asarray(self.arr[k]) for k in ("joint_mask", "act_mask", "target_mask"))),
                                 "context_digest": _digest(np.asarray(self.arr["move"]))}
        (self.out / "IDENTITY.json").write_text(json.dumps(ident, indent=1, sort_keys=True), encoding="utf-8")
        return ident


class TeacherShard:
    """Read-only view of a written shard; every batch is checked against the student's rows, masks, contexts."""

    def __init__(self, path: Path, expect: dict | None = None):
        self.path = Path(path)
        self.identity = json.loads((self.path / "IDENTITY.json").read_text(encoding="utf-8"))
        for k, v in (expect or {}).items():
            if self.identity.get(k) != v:
                raise ValueError(f"teacher shard {k} is {self.identity.get(k)!r}, expected {v!r}")
        self.arr = {k: np.load(self.path / f"{k}.npy", mmap_mode="r") for k in
                    ("row", "joint", "act", "target", "joint_mask", "act_mask", "target_mask", "move")}

    def batch(self, start: int, row_ids: np.ndarray, X: PolicyBatch, move: np.ndarray) -> dict[str, np.ndarray]:
        sl = slice(start, start + len(row_ids))
        if not np.array_equal(self.arr["row"][sl], row_ids):
            raise ValueError("row-identity mismatch: the teacher shard's rows are not the student batch's rows")
        if not np.array_equal(self.arr["move"][sl], move):
            raise ValueError("movement-context mismatch: teacher and student must use identical moves")
        jm, am, tm = masks(X, move)
        for k, m in (("joint_mask", jm), ("act_mask", am), ("target_mask", tm)):
            if not np.array_equal(self.arr[k][sl], m):
                raise ValueError(f"{k} mismatch between the teacher shard and the student batch")
        return {k: np.asarray(self.arr[k][sl]) for k in ("joint", "act", "target")}


# ------------------------------------------------------------------------------------------------- losses


def kl_term(student_logits: np.ndarray, teacher_p: np.ndarray, mask: np.ndarray, weight: np.ndarray, tau: float):
    """sum_rows w * tau^2 * KL(teacher || softmax(student / tau)) over masked entries, and its gradient w.r.t.
    the student logits: w * tau * (softmax(student / tau) - teacher). Rows with an empty mask contribute 0."""
    logq = masked_log_softmax(student_logits / tau, mask)
    q = np.where(mask, np.exp(logq), 0.0)
    has = mask.any(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(mask & (teacher_p > 0), np.log(np.where(teacher_p > 0, teacher_p, 1.0)), 0.0)
        kl = np.where(mask & (teacher_p > 0), teacher_p * (logp - np.where(mask, logq, 0.0)), 0.0).sum(-1)
    w = np.where(has, weight, 0.0)
    loss = float((w * tau * tau * kl).sum())
    grad = (w * tau)[..., None] * np.where(mask, q - teacher_p, 0.0)
    return loss, grad


def distill_loss(student: PolicyNet, X: PolicyBatch, y: dict, w: np.ndarray, teacher: dict | None, cfg: DistillConfig,
                 backward: bool) -> dict:
    """alpha * the trainer's CE loss + (1 - alpha) * tau^2 * KL, per head, weighted means; accumulates gradients.
    teacher None: the supervised loss alone (alpha = 1), the warmup stage."""
    cfg.check()
    assert not (backward and student.calibrated), "train with raw logits"
    a, tau = (cfg.alpha, cfg.tau) if teacher is not None else (1.0, cfg.tau)
    if teacher is None:
        teacher = {"joint": np.zeros(X.joint_mask.shape, np.float32), "act": np.zeros((len(w), 3, 4), np.float32),
                   "target": np.zeros(X.cand_valid.shape, np.float32)}
    wsum = max(float(w.sum()), 1e-12)
    out = {}
    move = y["y_move"].astype(np.int64)
    zj = student.joint_logits(X, keep=backward)
    jm = X.joint_mask
    lce, gce = softmax_xent(zj, jm, y["y_joint"].astype(np.int64), w, cfg.smoothing)
    lkl, gkl = kl_term(zj, teacher["joint"], jm, w, tau)
    out["joint"] = (a * lce + (1 - a) * lkl) / wsum
    if backward:
        gj = (a * gce + (1 - a) * gkl) / wsum
        D = gj.reshape(-1, 5, 5, 5)
        student.squad.backward(gj)
        student.mover.backward(np.stack([D.sum((2, 3)), D.sum((1, 3)), D.sum((1, 2))], axis=1)[..., None])
    wa = np.repeat(w, 3)
    za = student.act_logits(X, move, keep=backward)
    am = X.act_mask(move).reshape(-1, 4)
    lce, gce = softmax_xent(za.reshape(-1, 4), am, y["y_act"].astype(np.int64).reshape(-1), wa)
    lkl, gkl = kl_term(za.reshape(-1, 4), teacher["act"].reshape(-1, 4), am, wa, tau)
    out["act"] = (a * lce + (1 - a) * lkl) / wsum
    if backward:
        student.act.backward(cfg.act_weight * ((a * gce + (1 - a) * gkl) / wsum).reshape(za.shape))
    zt = student.target_logits(X, move, keep=backward)
    C = zt.shape[-1]
    tm = X.cand_valid.reshape(-1, C)
    lce, gce = softmax_xent(zt.reshape(-1, C), tm, y["y_tgt"].astype(np.int64).reshape(-1), wa)
    lkl, gkl = kl_term(zt.reshape(-1, C), teacher["target"].reshape(-1, C), tm, wa, tau)
    out["target"] = (a * lce + (1 - a) * lkl) / wsum
    if backward:
        student.target.backward(cfg.target_weight * ((a * gce + (1 - a) * gkl) / wsum).reshape(*zt.shape, 1))
    out["total"] = out["joint"] + cfg.act_weight * out["act"] + cfg.target_weight * out["target"]
    return out


# ------------------------------------------------------------------------------------------- 6XL estimates


def head_sizes(layout, family: str, hidden: dict) -> dict:
    """Exact layer dimensions, parameters and MACs per head for a hidden-width profile."""
    widths = policy_widths(layout, family)
    outs = {"mover": 1, "squad": JOINT, "act": 4, "target": 1}
    calls = {"mover": 15, "squad": 1, "act": 15, "target": 0}  # per opponent team turn (target: on demand)
    res = {}
    for h in ("mover", "squad", "act", "target"):
        dims = [widths[h], *hidden[h], outs[h]]
        params = sum(a * b + b for a, b in zip(dims[:-1], dims[1:]))
        macs = sum(a * b for a, b in zip(dims[:-1], dims[1:]))
        res[h] = {"dims": dims, "params": params, "macs_per_call": macs, "calls_per_team": calls[h],
                  "activations_per_row": sum(dims[1:])}
    return res


def teacher_estimates(layout, family: str = "p2", hidden: dict | None = None, batch: int = 4096) -> dict:
    """Memory of training a teacher profile (fp32 Adam): params, grads, two moments, and the forward activations
    kept for backward per batch (mover/act evaluated at 15 moves per row). Estimates, not a measured fit."""
    sizes = head_sizes(layout, family, hidden or TEACHER_6XL)
    params = sum(s["params"] for s in sizes.values())
    rows_per_batch = {"mover": 15 * batch, "squad": batch, "act": 3 * batch, "target": 3 * 8 * batch}
    act_floats = sum(2 * s["activations_per_row"] * rows_per_batch[h] for h, s in sizes.items())  # pre and post
    return {"heads": sizes, "params": params, "param_bytes": 4 * params, "adam_bytes": 4 * 3 * params,
            "activation_bytes_per_batch": 4 * act_floats, "batch": batch,
            "joint_macs_per_team": sum(s["macs_per_call"] * max(s["calls_per_team"], 0) for s in sizes.values()),
            "status": "estimate; the teacher is never embedded in a bot"}
