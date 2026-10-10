"""Policy and value networks over extractor records.

The head inputs are built exactly like inference/policy.hpp builds them:
    mover/act  [global, unit k, move features of (k, move)]
    squad      [global, unit 0..2, move features of every (k, move)]
    target     [dist, hp, carrying, support, hit after the move, attacked_by_team, target_home]
Family p3 (inference/policy_p3.hpp) appends the combat block of (k, move) to
mover/act and all 15 combat blocks to squad; the target head is unchanged.
Value family v2 reads the 54 legacy value features, v3 (inference/value_v3.hpp)
appends the value extension (fusion/families.py).
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


def policy_batch(rec: np.ndarray, layout: Layout, normalize: bool = True, family: str = "p2") -> PolicyBatch:
    if family not in ("p2", "p3"):
        raise ValueError(f"not a policy family: {family}")
    dtype = np.float32 if normalize else np.int64
    g = layout.get(rec, "g").astype(dtype)
    u = layout.get(rec, "u").astype(dtype)
    m = layout.get(rec, "m").astype(dtype)
    cf = layout.get(rec, "cand_f").astype(dtype)
    c = layout.get(rec, "pc").astype(dtype) if family == "p3" else None
    if normalize:
        g = g / layout.scale("global").astype(dtype)
        u = u / layout.scale("unit").astype(dtype)
        m = m / layout.scale("move").astype(dtype)
        cf = cf / layout.scale("cand_f").astype(dtype)
        if c is not None:
            c = c / layout.scale("combat").astype(dtype)
    B = len(rec)
    mover_parts = [np.broadcast_to(g[:, None, None, :], (B, 3, 5, g.shape[1])),
                   np.broadcast_to(u[:, :, None, :], (B, 3, 5, u.shape[2])), m]
    squad_parts = [g, u.reshape(B, -1), m.reshape(B, -1)]
    if c is not None:
        mover_parts.append(c)
        squad_parts.append(c.reshape(B, -1))
    mover_in = np.concatenate(mover_parts, axis=-1)
    squad_in = np.concatenate(squad_parts, axis=-1)
    return PolicyBatch(np.ascontiguousarray(mover_in), squad_in, layout.get(rec, "legal").astype(bool),
                       layout.get(rec, "act_legal").astype(bool), cf, layout.get(rec, "cand") >= 0)


def value_columns(rec: np.ndarray, layout: Layout, family: str = "v2") -> np.ndarray:
    """The raw int16 value inputs of a family (value_inputs without the scaling): what a V-only cache stores."""
    parts = [layout.get(rec, "v")] + ([layout.get(rec, "vx")] if family == "v3" else [])
    return np.ascontiguousarray(np.concatenate(parts, axis=-1))


def value_inputs(rec: np.ndarray, layout: Layout, normalize: bool = True, family: str = "v2") -> np.ndarray:
    v = layout.get(rec, "v")
    if family == "v3":
        v = np.concatenate([v, layout.get(rec, "vx")], axis=-1)
    elif family != "v2":
        raise ValueError(f"not a value family: {family}")
    return (v / layout.value_scale(family)).astype(np.float32) if normalize else v.astype(np.int64)


def policy_widths(layout: Layout, family: str = "p2") -> dict[str, int]:
    """Input width of each policy head (inference/policy.hpp kMoverIn / kSquadIn, policy_p3.hpp kMoverInP3 / kSquadInP3)."""
    G, U, M = len(layout.names("global")), len(layout.names("unit")), len(layout.names("move"))
    C = len(layout.names("combat")) if family == "p3" else 0
    return {"mover": G + U + M + C, "squad": G + 3 * U + 15 * M + 15 * C, "act": G + U + M + C, "target": len(TARGET_COLS)}


def combine_joint(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Joint logits from per-unit move scores s (B, 3, 5) and the squad residual r (B, 125).

    Summation order ((s0 + s1) + s2) + r matches inference/policy.hpp, so the
    double results are identical bit for bit.
    """
    j = (s[:, 0, :, None, None] + s[:, 1, None, :, None]) + s[:, 2, None, None, :]
    return j.reshape(-1, JOINT) + r


class PolicyNet:
    def __init__(self, layout: Layout, rng: np.random.Generator, hidden: dict | None = None, family: str = "p2"):
        hidden = hidden or {"mover": [16], "squad": [64, 32], "act": [16], "target": [8]}
        width = policy_widths(layout, family)
        self.hidden = hidden
        self.family = family
        self.mover = MLP([width["mover"], *hidden["mover"], 1], rng)
        self.squad = MLP([width["squad"], *hidden["squad"], JOINT], rng)
        self.act = MLP([width["act"], *hidden["act"], 4], rng)
        self.target = MLP([width["target"], *hidden["target"], 1], rng)
        self.beta = {"joint": 1.0, "act": 1.0, "target": 1.0}
        self.calibrated = False  # True: heads return beta * logits (evaluation and export)

    def batch(self, rec: np.ndarray, layout: Layout, normalize: bool = True) -> PolicyBatch:
        return policy_batch(rec, layout, normalize, self.family)

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

    def __init__(self, layout: Layout, rng: np.random.Generator, hidden=(128, 16), family: str = "v2"):
        self.hidden = list(hidden)
        self.family = family
        self.net = MLP([len(layout.value_scale(family)), *hidden, len(self.HEADS)], rng)

    def inputs(self, rec: np.ndarray, layout: Layout, normalize: bool = True) -> np.ndarray:
        return value_inputs(rec, layout, normalize, self.family)

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

    def rank_loss(self, xa, xb, y, w, cfg: dict, backward: bool) -> float:
        """Pairwise logistic loss on the outcome logit, P(a better than b) = sigmoid(z_a - z_b), weighted mean;
        accumulates rank_weight x its gradient (both states in one forward pass, so one backward)."""
        wsum = max(float(np.sum(w)), 1e-12)
        B = len(xa)
        z = self.logits(np.concatenate([xa, xb]), keep=backward)[:, 0]
        loss, g = bce_with_logits(z[:B] - z[B:], y, w)
        if backward:
            grad = np.zeros((2 * B, len(self.HEADS)), np.float32)
            grad[:B, 0], grad[B:, 0] = g, -g
            self.net.backward(cfg.get("rank_weight", 1.0) * grad / wsum)
        return loss / wsum

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
    """Python emulation of an exported model (inference/models/weights_<model>.npz), bit-exact with the C++."""

    NETS = ("mover", "squad", "act", "target", "value")

    def __init__(self, arrays: dict, policy_family: str | None = None, value_family: str | None = None, layout: Layout | None = None):
        self.q = {n: quant.unflatten(arrays, n) for n in self.NETS}
        mover_in, value_in = self.q["mover"]["in"], self.q["value"]["in"]
        if layout is not None:  # infer the families from the input widths, and check given ones against them
            by_width = {policy_widths(layout, f)["mover"]: f for f in ("p2", "p3")}
            v_width = {len(layout.value_scale(f)): f for f in ("v2", "v3") if all(g in layout.specs for g in layout.value_groups(f))}
            policy_family = policy_family or by_width.get(mover_in)
            value_family = value_family or v_width.get(value_in)
            if policy_widths(layout, policy_family or "p2")["mover"] != mover_in:
                raise ValueError(f"mover input {mover_in} does not match policy family {policy_family}")
            if len(layout.value_scale(value_family or "v2")) != value_in:
                raise ValueError(f"value input {value_in} does not match value family {value_family}")
        self.policy_family = policy_family or "p2"
        self.value_family = value_family or "v2"

    def raw(self, rec: np.ndarray, layout: Layout) -> np.ndarray:
        """Accumulators in the record layout of tools/cpp/infer_check.cpp, (R, 321) int64."""
        X = policy_batch(rec, layout, normalize=False, family=self.policy_family)
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
        value = quant.forward_raw(self.q["value"], value_inputs(rec, layout, normalize=False, family=self.value_family))
        return np.concatenate([mover.reshape(R, -1), squad, act.reshape(R, -1), target.reshape(R, -1), value], axis=1)

    def joint_logits(self, rec: np.ndarray, layout: Layout) -> tuple[np.ndarray, np.ndarray]:
        X = policy_batch(rec, layout, normalize=False, family=self.policy_family)
        s = quant.to_real(self.q["mover"], quant.forward_raw(self.q["mover"], X.mover_in)[..., 0])
        r = quant.to_real(self.q["squad"], quant.forward_raw(self.q["squad"], X.squad_in))
        return combine_joint(s, r), X.joint_mask

    def value_prob(self, rec: np.ndarray, layout: Layout) -> np.ndarray:
        z = quant.to_real(self.q["value"], quant.forward_raw(self.q["value"], value_inputs(rec, layout, normalize=False,
                                                                                         family=self.value_family)))[:, 0]
        return 1.0 / (1.0 + np.exp(-z))
