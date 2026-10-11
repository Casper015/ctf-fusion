"""Quantization-aware training (QAT) of low-bit P on the optional Torch backend (task T0030, docs/P_MIXED_PRECISION.md).

Forward values are the deployment's, exactly: every step quantizes the current float weights with the max-abs
rule of fusion/precision.py (a Torch port of quantize_layer, operation for operation in float64), then runs the
integer pipeline (int products summed in float64, exact below 2^53 since every sum is bounded by 2^30;
requantization in int64). Gradients flow through a float surrogate with the same weights rounded
(straight-through estimator): w_ste = w + (w_q - w).detach(), clipped ReLU with gradient only inside (0, 1),
and h = h_surrogate + (h_exact - h_surrogate).detach(). So "fake quantization" reproduces deployment scales,
clamps, rounding and requantization rather than approximating them, and tests compare it bit for bit with the
NumPy emulation.

Inputs are raw integer feature batches (fusion.models.policy_batch(..., normalize=False)): the deployed first
layer multiplies raw integers by weights with the input scales folded in, and so does QAT. The temperature
folded into the last layer at export (beta) is fitted after QAT on validation rows; QAT trains raw logits.

Loss (qat_loss): alpha * the trainer's cross-entropy + (1 - alpha) * tau^2 * KL(teacher || student / tau), per
head, as fusion/policy_distill.py. Nothing here trains a real model: tests run toy batches on the CPU.

FloatPolicy is the same loss on the float student (the warmup and distill stages): normalized features built on
the device from raw int16 records (fusion.torch_backend.DevicePolicyBatch, as the GPU trainer does), with the
NumPy model's arrays as starting weights and copied back by to_ref.
"""
from __future__ import annotations

import numpy as np
import torch

from . import precision
from .torch_backend import DevicePolicyBatch, TorchMLP, TorchPolicy, _ClippedReLU, _SoftmaxXent, combine_joint

F64 = torch.float64


def _round(x):
    return torch.floor(x + 0.5)


def quantize_layer(W: torch.Tensor, b: torch.Tensor, bits: int, first: bool, last: bool, x_max: torch.Tensor,
                   block: int = precision.BLOCK) -> dict:
    """fusion.precision.quantize_layer (max-abs, no importance) in Torch float64/int64. W: (out, in) folded."""
    W, b = W.to(F64), b.to(F64)
    out, n_in = W.shape
    lo, hi = precision.QRANGE[bits]
    nb = -(-n_in // block)
    cols = torch.arange(n_in, device=W.device) // block
    s_in = 1.0 if first else 1.0 / precision.ACT_ONE
    absw = W.abs()
    one = torch.ones((), dtype=F64, device=W.device)
    if bits != 4:
        peak = absw.amax(dim=1)
        S = torch.where(peak > 0, peak / hi, one)
        d = torch.ones((out, nb), dtype=torch.int64, device=W.device)
    else:
        peak_b = torch.stack([absw[:, i * block:(i + 1) * block].amax(dim=1) for i in range(nb)], dim=1)
        r = peak_b / hi
        rmax = r.amax(dim=1)
        S = torch.where(rmax > 0, rmax / precision.D_MAX, one)
        d = torch.clamp(_round(r / S[:, None]), 1, precision.D_MAX).to(torch.int64)
    q = torch.clamp(_round(W / (S[:, None] * d[:, cols])), lo, hi).to(torch.int64)
    for _ in range(40):
        B = _round(b / (S * s_in)).to(torch.int64)
        w_eff = d[:, cols] * q
        bound = (w_eff.abs() * x_max[None, :]).sum(dim=1) + B.abs()
        bad = bound > precision.ACC_LIMIT
        if not bool(bad.any()):
            break
        S = torch.where(bad, S * 2.0, S)
        q = torch.clamp(_round(W / (S[:, None] * d[:, cols])), lo, hi).to(torch.int64)
    else:
        raise ValueError("layer cannot be bounded inside the int32 accumulator range")
    L = {"bits": bits, "S": S, "d": d, "q": q, "B": B, "cols": cols, "s_in": s_in, "hidden": not last}
    if last:
        L["out_scale"] = S * s_in
    else:
        m = precision.ACT_ONE * S * s_in
        R = precision.R_MAX
        while R > 0 and bool((_round(m * 2.0 ** R) > 2 ** 31 - 1).any()):
            R -= 1
        L["mult"], L["rshift"] = _round(m * 2.0 ** R).to(torch.int64), R
    return L


def exact_layer(L: dict, h: torch.Tensor) -> torch.Tensor:
    """Deployment arithmetic of one quantized layer on integer inputs h (int64): next activations (int64) or,
    for the last layer, the real outputs (float64)."""
    w_eff = (L["d"][:, L["cols"]] * L["q"]).to(F64)
    acc = torch.round(h.to(F64) @ w_eff.T).to(torch.int64) + L["B"]     # exact: |every sum| <= 2^30 < 2^53
    if not L["hidden"]:
        return acc.to(F64) * L["out_scale"]
    R = L["rshift"]
    v = acc * L["mult"] + (1 << (R - 1) if R > 0 else 0)
    return torch.clamp(torch.div(v, 1 << R, rounding_mode="floor"), 0, precision.ACT_ONE)


class QATMLP:
    """A Torch MLP (fusion.torch_backend.TorchMLP parameters) whose forward is the deployed low-bit net."""

    def __init__(self, mlp, dev, bits: list[int], in_scale: np.ndarray, in_lo: np.ndarray, in_hi: np.ndarray):
        self.t = TorchMLP(mlp, dev)
        self.bits = bits
        self.in_scale = torch.tensor(np.asarray(in_scale, np.float64), device=dev)
        x_max = np.maximum(np.abs(np.asarray(in_lo)), np.abs(np.asarray(in_hi))).astype(np.int64)
        self.x_max = [torch.tensor(x_max, device=dev)] + [torch.full((w.shape[1],), precision.ACT_ONE, dtype=torch.int64,
                                                                     device=dev) for w in mlp.W[1:]]

    @property
    def params(self):
        return self.t.params

    def quantized(self) -> list[dict]:
        n = len(self.t.W)
        out = []
        for i, (W, b) in enumerate(zip(self.t.W, self.t.b)):
            Wf = W.detach().to(F64) / self.in_scale[None, :] if i == 0 else W.detach().to(F64)
            out.append(quantize_layer(Wf, b.detach(), self.bits[i], i == 0, i == n - 1, self.x_max[i]))
        return out

    def forward(self, x_raw: torch.Tensor, exact_only: bool = False):
        """x_raw: (..., in) raw integers. Returns float32 outputs whose values are the deployed outputs (float64
        exact values if exact_only) and whose gradient is the surrogate's."""
        lead = x_raw.shape[:-1]
        h_int = x_raw.reshape(-1, x_raw.shape[-1]).to(torch.int64)
        layers = self.quantized()
        if exact_only:
            for L in layers:
                h_int = exact_layer(L, h_int)
            return h_int.reshape(*lead, h_int.shape[-1])
        h = h_int.to(torch.float32)
        n = len(layers)
        for i, (L, W, b) in enumerate(zip(layers, self.t.W, self.t.b)):
            Wf = W / self.in_scale[None, :].to(W.dtype) if i == 0 else W
            Wq = (L["S"][:, None] * L["d"][:, L["cols"]].to(F64) * L["q"].to(F64)).to(W.dtype)
            W_ste = Wf + (Wq - Wf).detach()
            z = (h * L["s_in"]) @ W_ste.T + b
            exact = exact_layer(L, h_int)
            if i == n - 1:
                y = z + (exact.to(z.dtype) - z).detach()
                return y.reshape(*lead, y.shape[-1])
            surrogate = _ClippedReLU.apply(z) * precision.ACT_ONE
            h = surrogate + (exact.to(surrogate.dtype) - surrogate).detach()
            h_int = exact
        raise AssertionError("unreachable")


class QATPolicy:
    """The four policy heads under one precision profile (bits per layer from fusion.precision.PROFILES)."""

    def __init__(self, model, layout, profile: str, dev, profiles: dict | None = None):
        from training.export_weights import input_specs  # noqa: PLC0415

        prof = (profiles or precision.PROFILES)[profile]
        if prof.get("legacy"):
            raise ValueError("reference16 has its own (power-of-two) INT16 export; QAT targets the low-bit profiles")
        spec = input_specs(layout, model.family)
        self.model, self.profile = model, profile
        self.heads = {}
        for h in precision.HEADS:
            mlp = getattr(model, h)
            bits = [precision.layer_bits(prof, h, i, len(mlp.W)) for i in range(len(mlp.W))]
            self.heads[h] = QATMLP(mlp, dev, bits, *spec[h])

    @property
    def params(self):
        return [p for q in self.heads.values() for p in q.params]

    def logits(self, X, move: torch.Tensor):
        """(joint (B,125), act (B,3,4), target (B,3,C)) on a raw-integer batch X (fusion.models.PolicyBatch
        converted to tensors by `tensors`)."""
        s = self.heads["mover"].forward(X["mover_in"])[..., 0]
        r = self.heads["squad"].forward(X["squad_in"])
        b = torch.arange(move.shape[0], device=move.device)[:, None]
        k = torch.arange(3, device=move.device)[None, :]
        za = self.heads["act"].forward(X["mover_in"][b, k, move])
        zt = self.heads["target"].forward(target_in(X, move))[..., 0]
        return combine_joint(s, r), za, zt

    def to_ref(self):
        for q in self.heads.values():
            q.t.to_ref()

    def from_ref(self):
        for q in self.heads.values():
            q.t.from_ref()


class FloatPolicy:
    """The float student on a device (warmup and distill stages): logits(X, move) like QATPolicy, on batches from
    `device_batch`, so qat_loss computes the same alpha * CE + (1 - alpha) * tau^2 * KL as
    fusion.policy_distill.distill_loss."""

    def __init__(self, model, dev):
        self.t = TorchPolicy(model, dev)

    @property
    def params(self):
        return self.t.params

    def logits(self, X, move: torch.Tensor):
        n = self.t.nets
        s = n["mover"].forward(X["mover_in"])[..., 0]
        r = n["squad"].forward(X["squad_in"])
        b = torch.arange(move.shape[0], device=move.device)[:, None]
        k = torch.arange(3, device=move.device)[None, :]
        za = n["act"].forward(X["mover_in"][b, k, move])
        zt = n["target"].forward(target_in(X, move))[..., 0]
        return combine_joint(s, r), za, zt

    def to_ref(self):
        self.t.to_ref()

    def from_ref(self):
        self.t.from_ref()


def device_batch(rec: torch.Tensor, dl, family: str) -> dict:
    """Normalized features of raw int16 records already on the device (fusion.models.policy_batch(..., True))."""
    X = DevicePolicyBatch(rec, dl, family)
    return {"mover_in": X.mover_in, "squad_in": X.squad_in, "cand_f": X.cand_f, "joint_mask": X.joint_mask,
            "act_legal": X.act_legal, "cand_valid": X.cand_valid}


def tensors(X, dev) -> dict:
    """A raw-integer fusion.models.PolicyBatch as device tensors."""
    return {"mover_in": torch.tensor(np.asarray(X.mover_in, np.int64), device=dev),
            "squad_in": torch.tensor(np.asarray(X.squad_in, np.int64), device=dev),
            "cand_f": torch.tensor(np.asarray(X.cand_f, np.int64), device=dev),
            "joint_mask": torch.tensor(X.joint_mask, device=dev),
            "act_legal": torch.tensor(X.act_legal, device=dev),
            "cand_valid": torch.tensor(X.cand_valid, device=dev)}


def target_in(X: dict, move: torch.Tensor) -> torch.Tensor:
    from .models import TARGET_COLS  # noqa: PLC0415

    cf = X["cand_f"]
    idx = move[:, :, None, None].expand(cf.shape[0], 3, cf.shape[2], 1)
    hit = torch.gather(cf[..., 4:9], -1, idx)[..., 0]
    return torch.stack([hit if c is None else cf[..., c] for c in TARGET_COLS], dim=-1)


def kl_term(z: torch.Tensor, p: torch.Tensor, mask: torch.Tensor, w: torch.Tensor, tau: float) -> torch.Tensor:
    """sum w * tau^2 * KL(p || softmax(z / tau)) over masked entries (rows with an empty mask: 0)."""
    neg = torch.tensor(float("-inf"), dtype=z.dtype, device=z.device)
    has = mask.any(-1)
    zz = torch.where(mask, z / tau, neg)
    zz = torch.where(has[..., None], zz, torch.zeros_like(zz))
    logq = torch.log_softmax(zz, dim=-1)
    pos = mask & (p > 0)
    logp = torch.log(torch.where(pos, p, torch.ones_like(p)))
    kl = torch.where(pos, p * (logp - torch.where(mask, logq, torch.zeros_like(logq))), torch.zeros_like(p)).sum(-1)
    return (torch.where(has, w, torch.zeros_like(w)) * tau * tau * kl).sum()


def qat_loss(policy: QATPolicy, X: dict, y: dict, w: torch.Tensor, teacher: dict | None, cfg) -> tuple[torch.Tensor, dict]:
    """alpha * CE + (1 - alpha) * tau^2 * KL per head (teacher None: CE only), weighted means, head weights."""
    a = cfg.alpha if teacher is not None else 1.0
    tau = cfg.tau
    wsum = w.sum().clamp(min=1e-12)
    move = y["y_move"]
    zj, za, zt = policy.logits(X, move)
    b = torch.arange(move.shape[0], device=move.device)[:, None]
    k = torch.arange(3, device=move.device)[None, :]
    am = X["act_legal"][b, k, move]
    w3 = w.repeat_interleave(3)
    C = zt.shape[-1]
    parts = {}
    for name, z, mask, label, ww, tp in (
            ("joint", zj, X["joint_mask"], y["y_joint"], w, None if teacher is None else teacher["joint"]),
            ("act", za.reshape(-1, 4), am.reshape(-1, 4), y["y_act"].reshape(-1), w3,
             None if teacher is None else teacher["act"].reshape(-1, 4)),
            ("target", zt.reshape(-1, C), X["cand_valid"].reshape(-1, C), y["y_tgt"].reshape(-1), w3,
             None if teacher is None else teacher["target"].reshape(-1, C))):
        ce = _SoftmaxXent.apply(z, mask, label, ww, float(cfg.smoothing if name == "joint" else 0.0))
        loss = a * ce
        if tp is not None and a < 1.0:
            loss = loss + (1 - a) * kl_term(z, tp, mask, ww, tau)
        parts[name] = loss / wsum
    total = parts["joint"] + cfg.act_weight * parts["act"] + cfg.target_weight * parts["target"]
    return total, {k: v.detach() for k, v in parts.items()} | {"total": total.detach()}


__all__ = ["quantize_layer", "exact_layer", "QATMLP", "QATPolicy", "FloatPolicy", "device_batch", "tensors", "qat_loss",
           "kl_term"]
