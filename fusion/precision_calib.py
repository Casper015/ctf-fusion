"""Activation-aware calibration and sensitivity diagnostics for low-bit P (task T0030, docs/P_MIXED_PRECISION.md).

Borrowed from llama.cpp's importance matrix (tools/imatrix) and AWQ: a weight matters in proportion to how large
its input usually is. ActivationStats streams, per head and layer, the second moment E[x_i^2] of every layer
input over bounded batches: raw integer features for first layers (the folded weights multiply raw integers),
clipped-ReLU activations on the 0..128 scale for hidden layers. The importance vector weights the scale search
in fusion/precision.py (sum_i imp_i (w_i - w_eff_i)^2) and the reconstruction error reported here.

Diagnostics, all on declared calibration rows (train or validation; a test or confirmation split is refused):
  reconstruction   importance-weighted weight error per layer
  distributions    KL(reference || profile) of the masked joint movement, and of act and target under the same
                   movement contexts (the recorded moves), overall and per rare context
  sensitivity      one layer at a time fake-quantized (others float): the change it alone causes, to rank layers
                   for protection (whole layers or contiguous blocks: no sparse per-weight exception path)
  coverage         rows of each rare context: carrier, low HP, pickup, drop, delivery, combat

Rows are processed in batches; memory is bounded by the batch, never by the dataset.
"""
from __future__ import annotations

import copy

import numpy as np

from . import precision
from .models import PolicyBatch, PolicyNet, policy_batch
from .nn import masked_log_softmax

FORBIDDEN_SPLITS = ("test", "confirm", "confirmation", "holdout", "ood")
LOW_HP = 34          # one hit (damage 34) from death at hp 100 - 2*33: "low" means a single hit kills
CONTEXTS = ("all", "carrier", "low_hp", "pickup", "drop", "delivery", "combat")


def check_split(split: str) -> None:
    if any(f in split.lower() for f in FORBIDDEN_SPLITS):
        raise ValueError(f"calibration refuses the {split!r} rows: scales and thresholds are fitted on train or validation only")


def _layer_inputs(mlp, x: np.ndarray) -> list[np.ndarray]:
    """Inputs of every layer of a float MLP on normalized x: [x, h1, h2, ...] (h in [0, 1])."""
    h = x.reshape(-1, x.shape[-1]).astype(np.float64)
    outs = [h]
    for i, (w, b) in enumerate(zip(mlp.W, mlp.b)):
        z = h @ w.T.astype(np.float64) + b
        if i < len(mlp.W) - 1:
            h = np.clip(z, 0.0, 1.0)
            outs.append(h)
    return outs


class ActivationStats:
    """Streaming E[x^2], max |x| and row counts per head and layer input, by source (recorded / successor)."""

    def __init__(self, policy: PolicyNet, split: str):
        check_split(split)
        self.policy = policy
        self.split = split
        self.sum2: dict[str, list[np.ndarray]] = {}
        self.peak: dict[str, list[np.ndarray]] = {}
        self.count: dict[str, int] = {}
        self.rows = {"recorded": 0, "successor": 0}

    def _add(self, head: str, layers: list[np.ndarray]):
        if head not in self.sum2:
            self.sum2[head] = [np.zeros(l.shape[1]) for l in layers]
            self.peak[head] = [np.zeros(l.shape[1]) for l in layers]
            self.count[head] = 0
        for i, l in enumerate(layers):
            self.sum2[head][i] += (l * l).sum(axis=0)
            self.peak[head][i] = np.maximum(self.peak[head][i], np.abs(l).max(axis=0, initial=0))
        self.count[head] += layers[0].shape[0]

    def update(self, Xn: PolicyBatch, Xr: PolicyBatch, move: np.ndarray, source: str = "recorded"):
        """Xn: normalized batch (float forward), Xr: the same rows as raw integers; move: (B, 3) contexts."""
        if source not in self.rows:
            raise ValueError(f"unknown calibration source {source!r}")
        self.rows[source] += len(move)
        p = self.policy
        legal = Xn.legal.reshape(-1)
        for head, xn, xr, keep in (("mover", Xn.mover_in, Xr.mover_in, legal),
                                   ("squad", Xn.squad_in, Xr.squad_in, None),
                                   ("act", Xn.act_in(move), Xr.act_in(move), None),
                                   ("target", Xn.target_in(move), Xr.target_in(move), Xn.cand_valid.reshape(-1))):
            mlp = getattr(p, head)
            ins = _layer_inputs(mlp, xn)
            ins[0] = xr.reshape(-1, xr.shape[-1]).astype(np.float64)              # raw integers: the folded scale
            ins[1:] = [h * precision.ACT_ONE for h in ins[1:]]                    # 0..128, as deployed
            if keep is not None:
                ins = [l[keep] for l in ins]
            self._add(head, ins)

    def importance(self, eps: float = 1e-6) -> dict[str, list[np.ndarray]]:
        """Per head and layer, E[x_i^2] scaled to mean 1 (+ eps so an unused input still has a little weight)."""
        out = {}
        for head, layers in self.sum2.items():
            n = max(self.count[head], 1)
            out[head] = []
            for s in layers:
                m = s / n
                out[head].append(m / max(m.mean(), 1e-12) + eps)
        return out

    def receipt(self) -> dict:
        return {"split": self.split, "rows": dict(self.rows), "inputs": {h: self.count[h] for h in self.count}}


def collect(policy: PolicyNet, layout, rec: np.ndarray, labels: dict, split: str, batch: int = 4096,
            max_rows: int | None = None, successors: np.ndarray | None = None) -> ActivationStats:
    """Stream calibration rows (and optional recorded/synthetic successor states, with their own move contexts
    in labels['successor_move']) through ActivationStats, `batch` rows at a time."""
    stats = ActivationStats(policy, split)
    n = len(rec) if max_rows is None else min(len(rec), max_rows)
    for a in range(0, n, batch):
        r = rec[a:min(n, a + batch)]
        mv = labels["y_move"][a:a + len(r)].astype(np.int64)
        stats.update(policy_batch(r, layout, True, policy.family), policy_batch(r, layout, False, policy.family), mv)
    if successors is not None:
        for a in range(0, len(successors), batch):
            r = successors[a:a + batch]
            mv = labels["successor_move"][a:a + len(r)].astype(np.int64)
            stats.update(policy_batch(r, layout, True, policy.family), policy_batch(r, layout, False, policy.family), mv,
                         "successor")
    return stats


# ------------------------------------------------------------------------------------------- diagnostics


def reconstruction(qp: dict, policy: PolicyNet, layout, importance: dict | None = None) -> dict:
    """Importance-weighted squared weight error per head and layer (folded weights, as quantized)."""
    from training.export_weights import input_specs  # noqa: PLC0415

    spec = input_specs(layout, policy.family)
    out = {}
    for h in precision.HEADS:
        mlp, net = getattr(policy, h), qp["nets"][h]
        rows = []
        for i, L in enumerate(net["layers"]):
            W = mlp.W[i].astype(np.float64)
            if i == 0:
                W = W / spec[h][0][None, :]
            if i == len(mlp.W) - 1:
                W = W * (policy.beta["act"] if h == "act" else policy.beta["target"] if h == "target" else policy.beta["joint"])
            cols = np.arange(L["in"]) // L["block"]
            Wq = L["row_scale"][:, None] * L["d"].astype(np.float64)[:, cols] * L["q"].astype(np.float64)
            imp = np.ones(L["in"]) if importance is None else importance[h][i]
            rows.append({"layer": i, "bits": L["bits"], "weighted_error": float(((W - Wq) ** 2 * imp[None, :]).sum()),
                         "relative": float(((W - Wq) ** 2 * imp[None, :]).sum() / max(((W ** 2) * imp[None, :]).sum(), 1e-30))})
        out[h] = rows
    return out


def _kl(logp_ref: np.ndarray, logp: np.ndarray, mask: np.ndarray) -> np.ndarray:
    p = np.where(mask, np.exp(logp_ref), 0.0)
    return np.where(mask, p * (np.where(mask, logp_ref, 0) - np.where(mask, logp, 0)), 0.0).sum(-1)


def contexts(Xr: PolicyBatch, layout, labels: dict) -> dict[str, np.ndarray]:
    """Boolean row masks of the rare contexts (raw-integer batch Xr, labels of the same rows)."""
    U = layout.names("unit")
    M = layout.names("move")
    u = Xr.mover_in[:, :, 0, len(layout.names("global")):len(layout.names("global")) + len(U)]  # (B, 3, U)
    carrying = u[..., U.index("carrying")] > 0
    hp = u[..., U.index("hp")]
    active = u[..., U.index("active")] > 0
    home = u[..., U.index("home_dist")]
    mv = Xr.mover_in[..., len(layout.names("global")) + len(U):len(layout.names("global")) + len(U) + len(M)]
    act = labels["y_act"].astype(np.int64)
    return {"all": np.ones(len(act), bool),
            "carrier": carrying.any(1),
            "low_hp": (active & (hp <= LOW_HP)).any(1),
            "pickup": (act == 1).any(1),
            "drop": (act == 2).any(1),
            "delivery": (carrying & (home <= 2)).any(1),
            "combat": (act == 3).any(1) | (mv[..., M.index("targets")] > 0).any((1, 2))}


def distributions(ref_logits, test_logits, Xr: PolicyBatch, labels: dict, layout) -> dict:
    """Mean KL(ref || test) of the masked joint, act and target heads, overall and per rare context.

    ref_logits / test_logits: callables (X, move) -> (joint (B,125), act (B,3,4), target (B,3,C)) on raw rows."""
    move = labels["y_move"].astype(np.int64)
    jr, ar, tr = ref_logits(Xr, move)
    jt, at, tt = test_logits(Xr, move)
    jm = Xr.joint_mask
    kj = _kl(masked_log_softmax(jr, jm), masked_log_softmax(jt, jm), jm)
    am = Xr.act_mask(move)
    ka = _kl(masked_log_softmax(ar, am), masked_log_softmax(at, am), am).mean(1)
    tm = Xr.cand_valid
    has_t = tm.any(-1)
    kt_u = np.where(has_t, _kl(masked_log_softmax(tr, tm), masked_log_softmax(tt, tm), tm), 0.0)
    kt = kt_u.sum(1) / np.maximum(has_t.sum(1), 1)
    out = {}
    for name, rows in contexts(Xr, layout, labels).items():
        n = int(rows.sum())
        out[name] = {"rows": n, "joint_kl": float(kj[rows].mean()) if n else None,
                     "act_kl": float(ka[rows].mean()) if n else None, "target_kl": float(kt[rows].mean()) if n else None}
    return out


def profile_logits(qp: dict):
    """(X raw, move) -> (joint, act, target) logits of an exported profile (integer emulation)."""
    def f(X: PolicyBatch, move):
        joint = precision.joint_logits(qp, X)
        act = precision.head_forward(qp, "act", X.act_in(move))
        tgt = precision.head_forward(qp, "target", X.target_in(move))[..., 0]
        return joint, act, tgt
    return f


def float_logits(policy: PolicyNet, layout, rec: np.ndarray):
    """The float model's calibrated logits on the same rows (the reference a profile is compared with)."""
    Xn = policy_batch(rec, layout, True, policy.family)

    def f(X: PolicyBatch, move):
        was = policy.calibrated
        policy.calibrated = True
        try:
            return (policy.joint_logits(Xn), policy.act_logits(Xn, move), policy.target_logits(Xn, move))
        finally:
            policy.calibrated = was
    return f


def fake_quant_layer(policy: PolicyNet, layout, head: str, index: int, bits: int, importance: dict | None = None) -> PolicyNet:
    """A copy of the float policy with one layer's weights replaced by their dequantized low-bit values."""
    from training.export_weights import input_specs  # noqa: PLC0415

    spec = input_specs(layout, policy.family)
    p = copy.deepcopy(policy)
    mlp = getattr(p, head)
    W = mlp.W[index].astype(np.float64)
    first = index == 0
    scale = spec[head][0] if first else np.ones(W.shape[1])
    Wf = W / scale[None, :]
    x_max = np.maximum(np.abs(spec[head][1]), np.abs(spec[head][2])).astype(np.int64) if first else np.full(W.shape[1], 128)
    imp = None if importance is None else importance[head][index]
    L = precision.quantize_layer(Wf, mlp.b[index], bits, first, index == len(mlp.W) - 1, x_max, imp)
    cols = np.arange(L["in"]) // L["block"]
    Wq = L["row_scale"][:, None] * L["d"].astype(np.float64)[:, cols] * L["q"].astype(np.float64)
    mlp.W[index][...] = (Wq * scale[None, :]).astype(mlp.W[index].dtype)
    return p


def sensitivity(policy: PolicyNet, layout, rec: np.ndarray, labels: dict, split: str, bits: int = 4,
                importance: dict | None = None) -> list[dict]:
    """Rank layers: the joint/act/target KL each one causes alone when fake-quantized to `bits` (others float)."""
    check_split(split)
    Xr = policy_batch(rec, layout, False, policy.family)
    ref = float_logits(policy, layout, rec)
    rows = []
    for h in precision.HEADS:
        for i in range(len(getattr(policy, h).W)):
            q = fake_quant_layer(policy, layout, h, i, bits, importance)
            d = distributions(ref, float_logits(q, layout, rec), Xr, labels, layout)["all"]
            rows.append({"head": h, "layer": i, "bits": bits, **{k: d[k] for k in ("joint_kl", "act_kl", "target_kl")}})
    return sorted(rows, key=lambda r: -(r["joint_kl"] + r["act_kl"] + r["target_kl"]))


def report(policy: PolicyNet, layout, rec: np.ndarray, labels: dict, split: str, profiles=("uniform8", "mixed4_8"),
           importance: dict | None = None) -> dict:
    """Profiles against the INT16 reference (reference16) on calibration rows: distributions, reconstruction,
    storage. A diagnostic of quantization error, not a strength claim."""
    check_split(split)
    Xr = policy_batch(rec, layout, False, policy.family)
    ref16 = precision.quantize_policy(policy, layout, "reference16")
    out = {"split": split, "rows": len(rec), "reference": "reference16", "profiles": {}}
    for name in profiles:
        qp = precision.quantize_policy(policy, layout, name, importance=importance)
        out["profiles"][name] = {
            "identity": precision.manifest_identity(qp)["weights_digest"],
            "distributions": distributions(profile_logits(ref16), profile_logits(qp), Xr, labels, layout),
            "reconstruction": reconstruction(qp, policy, layout, importance),
            "storage": {h: {k: v for k, v in s.items() if k != "layers"} for h, s in precision.policy_storage(qp).items()},
        }
    return out
