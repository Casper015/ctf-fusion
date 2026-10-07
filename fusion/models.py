"""Policy and value networks over extractor records.

The head inputs are built exactly like inference/policy.hpp builds them:
    mover/act  [global, unit k, move features of (k, move)]
    squad      [global, unit 0..2, move features of every (k, move)]
    target     [dist, hp, carrying, support, hit after the move, attacked_by_team, target_home]
Training feeds them divided by the spec scales; the exporter folds those
scales into the first layer so the C++ side consumes the raw integers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import quant
from .nn import MLP, bce_with_logits, softmax_xent
from .schema import Layout

TARGET_COLS = (0, 1, 2, 3, None, 9, 10)  # None = hit bit of the chosen move (cand_f[4 + move])
JOINT = 125


@dataclass
class PolicyBatch:
    mover_in: np.ndarray    # (B, 3, 5, G+U+M)
    squad_in: np.ndarray    # (B, G+3U+15M)
    legal: np.ndarray       # (B, 3, 5) bool
    act_legal: np.ndarray   # (B, 3, 5, 4) bool
    cand_f: np.ndarray      # (B, 3, C, TF)
    cand_valid: np.ndarray  # (B, 3, C) bool

    @property
    def joint_mask(self) -> np.ndarray:
        l = self.legal
        return (l[:, 0, :, None, None] & l[:, 1, None, :, None] & l[:, 2, None, None, :]).reshape(-1, JOINT)

    def act_in(self, move: np.ndarray) -> np.ndarray:
        """Act-head input for a chosen move per unit, move: (B, 3)."""
        b = np.arange(len(move))[:, None]
        return self.mover_in[b, np.arange(3)[None, :], move]

    def act_mask(self, move: np.ndarray) -> np.ndarray:
        b = np.arange(len(move))[:, None]
        return self.act_legal[b, np.arange(3)[None, :], move]

    def target_in(self, move: np.ndarray) -> np.ndarray:
        """Target-head input (B, 3, C, 7) for a chosen move per unit."""
        cf = self.cand_f
        hit = np.take_along_axis(cf[..., 4:9], move[:, :, None, None].astype(np.int64), axis=-1)[..., 0]
        cols = [hit if c is None else cf[..., c] for c in TARGET_COLS]
        return np.stack(cols, axis=-1)


def policy_batch(rec: np.ndarray, layout: Layout, normalize: bool = True) -> PolicyBatch:
    dtype = np.float32 if normalize else np.int64
    g = layout.get(rec, "g").astype(dtype)
    u = layout.get(rec, "u").astype(dtype)
    m = layout.get(rec, "m").astype(dtype)
    cf = layout.get(rec, "cand_f").astype(dtype)
    if normalize:
        g = g / layout.scale("global").astype(dtype)
        u = u / layout.scale("unit").astype(dtype)
        m = m / layout.scale("move").astype(dtype)
        cf = cf / layout.scale("cand_f").astype(dtype)
    B = len(rec)
    mover_in = np.concatenate([np.broadcast_to(g[:, None, None, :], (B, 3, 5, g.shape[1])),
                               np.broadcast_to(u[:, :, None, :], (B, 3, 5, u.shape[2])), m], axis=-1)
    squad_in = np.concatenate([g, u.reshape(B, -1), m.reshape(B, -1)], axis=-1)
    return PolicyBatch(np.ascontiguousarray(mover_in), squad_in, layout.get(rec, "legal").astype(bool),
                       layout.get(rec, "act_legal").astype(bool), cf, layout.get(rec, "cand") >= 0)


def value_inputs(rec: np.ndarray, layout: Layout, normalize: bool = True) -> np.ndarray:
    v = layout.get(rec, "v")
    return (v / layout.value_scale()).astype(np.float32) if normalize else v.astype(np.int64)


def combine_joint(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Joint logits from per-unit move scores s (B, 3, 5) and the squad residual r (B, 125).

    Summation order ((s0 + s1) + s2) + r matches inference/policy.hpp, so the
    double results are identical bit for bit.
    """
    j = (s[:, 0, :, None, None] + s[:, 1, None, :, None]) + s[:, 2, None, None, :]
    return j.reshape(-1, JOINT) + r


class PolicyNet:
    def __init__(self, layout: Layout, rng: np.random.Generator, hidden: dict | None = None):
        hidden = hidden or {"mover": [16], "squad": [64, 32], "act": [16], "target": [8]}
        G, U, M = len(layout.names("global")), len(layout.names("unit")), len(layout.names("move"))
        self.hidden = hidden
        self.mover = MLP([G + U + M, *hidden["mover"], 1], rng)
        self.squad = MLP([G + 3 * U + 15 * M, *hidden["squad"], JOINT], rng)
        self.act = MLP([G + U + M, *hidden["act"], 4], rng)
        self.target = MLP([len(TARGET_COLS), *hidden["target"], 1], rng)
        self.beta = {"joint": 1.0, "act": 1.0, "target": 1.0}
        self.calibrated = False  # True: heads return beta * logits (evaluation and export)

    @property
    def nets(self):
        return {"mover": self.mover, "squad": self.squad, "act": self.act, "target": self.target}

    @property
    def params(self):
        return [p for n in self.nets.values() for p in n.params]

    @property
    def grads(self):
        return [g for n in self.nets.values() for g in n.grads]

    def zero_grad(self):
        for n in self.nets.values():
            n.zero_grad()

    def _beta(self, head):
        return self.beta[head] if self.calibrated else 1.0

    def joint_logits(self, X: PolicyBatch, keep=False) -> np.ndarray:
        s = self.mover.forward(X.mover_in, keep)[..., 0]
        r = self.squad.forward(X.squad_in, keep)
        return self._beta("joint") * combine_joint(s, r)

    def act_logits(self, X: PolicyBatch, move, keep=False) -> np.ndarray:
        return self._beta("act") * self.act.forward(X.act_in(move), keep)

    def target_logits(self, X: PolicyBatch, move, keep=False) -> np.ndarray:
        return self._beta("target") * self.target.forward(X.target_in(move), keep)[..., 0]

    def loss(self, X: PolicyBatch, y: dict, w: np.ndarray, cfg: dict, backward: bool) -> dict:
        """Weighted mean losses of the three heads; accumulates gradients if `backward`."""
        assert not (backward and self.calibrated), "train with raw logits"
        wsum = max(float(w.sum()), 1e-12)
        out = {}
        logits = self.joint_logits(X, keep=backward)
        lj, gj = softmax_xent(logits, X.joint_mask, y["y_joint"].astype(np.int64), w, cfg.get("smoothing", 0.0))
        out["joint"] = lj / wsum
        if backward:
            gj = gj / wsum
            D = gj.reshape(-1, 5, 5, 5)
            ds = np.stack([D.sum((2, 3)), D.sum((1, 3)), D.sum((1, 2))], axis=1)
            self.squad.backward(gj)
            self.mover.backward(ds[..., None])

        move = y["y_move"].astype(np.int64)
        ya = y["y_act"].astype(np.int64)
        za = self.act_logits(X, move, keep=backward)
        wa = np.repeat(w, 3)
        la, ga = softmax_xent(za.reshape(-1, 4), X.act_mask(move).reshape(-1, 4), ya.reshape(-1), wa)
        out["act"] = la / wsum
        if backward:
            self.act.backward(cfg.get("act_weight", 1.0) * ga.reshape(za.shape) / wsum)

        yt = y["y_tgt"].astype(np.int64)
        zt = self.target_logits(X, move, keep=backward)
        C = zt.shape[-1]
        lt, gt = softmax_xent(zt.reshape(-1, C), X.cand_valid.reshape(-1, C), yt.reshape(-1), wa)
        out["target"] = lt / wsum
        if backward:
            self.target.backward(cfg.get("target_weight", 1.0) * gt.reshape(*zt.shape, 1) / wsum)
        out["total"] = out["joint"] + cfg.get("act_weight", 1.0) * out["act"] + cfg.get("target_weight", 1.0) * out["target"]
        return out

    def state(self) -> dict:
        st = {f"{name}.{k}": v for name, net in self.nets.items() for k, v in net.state().items()}
        st |= {f"beta.{k}": np.array(v) for k, v in self.beta.items()}
        return st

    def load(self, st: dict):
        for name, net in self.nets.items():
            net.load({k.split(".", 1)[1]: v for k, v in st.items() if k.startswith(name + ".")})
        self.beta = {k: float(st[f"beta.{k}"]) for k in self.beta}


class ValueNet:
    HEADS = ("outcome", "capture_h", "death_h")

    def __init__(self, layout: Layout, rng: np.random.Generator, hidden=(128, 16)):
        self.hidden = list(hidden)
        self.net = MLP([len(layout.value_scale()), *hidden, len(self.HEADS)], rng)

    @property
    def params(self):
        return self.net.params

    @property
    def grads(self):
        return self.net.grads

    def zero_grad(self):
        self.net.zero_grad()

    def logits(self, x, keep=False):
        return self.net.forward(x, keep)

    def loss(self, x, y, w, cfg: dict, backward: bool) -> dict:
        wsum = max(float(w.sum()), 1e-12)
        head_w = np.array([1.0, cfg.get("aux_weight", 0.25), cfg.get("aux_weight", 0.25)], np.float32)
        z = self.logits(x, keep=backward)
        weight = w[:, None] * head_w[None, :]
        _, grad = bce_with_logits(z, y, weight)
        out = {h: bce_with_logits(z[:, i], y[:, i], w)[0] / wsum for i, h in enumerate(self.HEADS)}
        out["total"] = float(sum(head_w[i] * out[h] for i, h in enumerate(self.HEADS)))
        if backward:
            self.net.backward(grad / wsum)
        return out

    def state(self) -> dict:
        return self.net.state()

    def load(self, st: dict):
        self.net.load(st)


class QuantizedTier:
    """Python emulation of an exported tier (inference/models/<tier>.npz), bit-exact with the C++."""

    NETS = ("mover", "squad", "act", "target", "value")

    def __init__(self, arrays: dict):
        self.q = {n: quant.unflatten(arrays, n) for n in self.NETS}

    def raw(self, rec: np.ndarray, layout: Layout) -> np.ndarray:
        """Accumulators in the record layout of tools/infer_check.cpp, (R, 321) int64."""
        X = policy_batch(rec, layout, normalize=False)
        R = len(rec)
        legal = X.legal
        mover = quant.forward_raw(self.q["mover"], X.mover_in)[..., 0] * legal
        squad = quant.forward_raw(self.q["squad"], X.squad_in)
        act = quant.forward_raw(self.q["act"], X.mover_in) * legal[..., None]
        C = X.cand_f.shape[2]
        target = np.zeros((R, 3, C, 5), np.int64)
        for m in range(5):
            move = np.full((R, 3), m, np.int64)
            target[..., m] = quant.forward_raw(self.q["target"], X.target_in(move))[..., 0] * X.cand_valid
        value = quant.forward_raw(self.q["value"], value_inputs(rec, layout, normalize=False))
        return np.concatenate([mover.reshape(R, -1), squad, act.reshape(R, -1), target.reshape(R, -1), value], axis=1)

    def joint_logits(self, rec: np.ndarray, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
        X = policy_batch(rec, layout, normalize=False)
        s = quant.to_real(self.q["mover"], quant.forward_raw(self.q["mover"], X.mover_in)[..., 0])
        r = quant.to_real(self.q["squad"], quant.forward_raw(self.q["squad"], X.squad_in))
        return combine_joint(s, r), X.joint_mask

    def value_prob(self, rec: np.ndarray, layout: Layout) -> np.ndarray:
        z = quant.to_real(self.q["value"], quant.forward_raw(self.q["value"], value_inputs(rec, layout, normalize=False)))[:, 0]
        return 1.0 / (1.0 + np.exp(-z))
