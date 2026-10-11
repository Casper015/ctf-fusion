"""Experimental mixed-precision formats for the opponent policy P (task T0030, docs/P_MIXED_PRECISION.md).

Three profiles share the shapes and features of an exported policy:

    reference16   the existing INT16 export (fusion/quant.py), unchanged: power-of-two shifts, int16 weights
    uniform8      every layer int8, one scale per output row
    mixed4_8      int8, except the hidden matrices of the squad head: packed int4 with 6-bit block scales

A low-bit layer (bits 8 or 4, also 16 for a sensitive layer kept wide) is integer from input to accumulator:

    w_eff[o][i] = d[o][i / 32] * q[o][i]       q in [-127, 127] (8), [-8, 7] (4), [-32767, 32767] (16); d = 1
                                               except for 4 bits, where d in [1, 63] is a per-block integer scale
    acc[o]      = B[o] + sum_i w_eff[o][i] * x[i]                    int32: |acc| <= bound <= 2^30, proven here
    real z[o]   = S[o] * s_in * acc[o]         S: row scale (float64); s_in = 1 for raw features, 1/128 for hidden
    hidden:   h = clamp((acc * M[o] + 2^(R-1)) >> R, 0, 128)          M/2^R ~ 128 S s_in; acc * M < 2^62 (int64)
    output:   real = double(acc) * out_scale[o]                       out_scale = S s_in, an exact float64

Hidden activations stay on the existing clipped-ReLU scale 0..128 (they fit uint8, not int8). First-layer
inputs are the extractor's signed integers and travel as int16, as in the INT16 path. Every intermediate sum of
a kernel (a block's partial sum, d times it, any order of additions) is bounded by the same per-row bound, so
integer addition cannot overflow and every kernel path gives identical bits.

Calibration (fusion/precision_calib.py) supplies per-input importance (activation second moments, the IQ/AWQ
idea): the scale search then minimizes sum_i imp_i (w_i - w_eff_i)^2 instead of the plain maximum rule. With no
importance (QAT, fusion/torch_qat.py) the deterministic max-abs rule is used, which the Torch port reproduces
operation by operation.

Nothing here changes an existing artifact: exports of these profiles go under build/ (training/export_precision.py).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import quant
from .nn import MLP

FORMAT_VERSION = "pprec-1"
BLOCK = 32                      # input columns per block (int4 sub-scale, packing and SIMD unit)
ACT_ONE = quant.ACT_ONE         # 128
ACC_LIMIT = quant.ACC_LIMIT     # 2^30
QRANGE = {16: (-32767, 32767), 8: (-127, 127), 4: (-8, 7)}
D_MAX = 63                      # 6-bit block scale (k-quant style)
R_MAX = 46                      # requantization shift cap
SCALE_GRID = tuple(np.round(np.linspace(0.70, 1.10, 21), 4))  # weighted search: multiples of the max-abs scale
HEADS = ("mover", "squad", "act", "target")

# Built-in profiles; profiles/p_precision_code.json mirrors them (a test keeps the two equal).
PROFILES = {
    "reference16": {"legacy": True, "default": 16, "rules": []},
    "uniform8": {"legacy": False, "default": 8, "rules": []},
    "mixed4_8": {"legacy": False, "default": 8, "rules": [{"head": "squad", "layers": "hidden", "bits": 4}]},
}


def layer_kind(index: int, count: int) -> str:
    return "first" if index == 0 else "last" if index == count - 1 else "hidden"


def layer_bits(profile: dict, head: str, index: int, count: int) -> int:
    """The weight bits of one layer under a profile: the last matching rule, else the default."""
    bits = profile["default"]
    kind = layer_kind(index, count)
    for r in profile.get("rules", []):
        if r["head"] in (head, "*") and r["layers"] in (kind, "*", index):
            bits = r["bits"]
    if bits not in QRANGE:
        raise ValueError(f"{head} layer {index}: unsupported weight bits {bits}")
    return bits


def profile_identity(name: str, profile: dict, block: int = BLOCK) -> str:
    """Hash of everything that defines a profile's numerics (not the weights)."""
    body = json.dumps({"format": FORMAT_VERSION, "name": name, "block": block, "profile": profile}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def load_profiles(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: {"legacy": v.get("legacy", False), "default": v["default"], "rules": v.get("rules", [])}
            for k, v in cfg["profiles"].items()}


def _round(x) -> np.ndarray:
    return np.floor(np.asarray(x, np.float64) + 0.5)


@dataclass
class _Fit:
    S: np.ndarray       # (out,) float64 row scale
    d: np.ndarray       # (out, nb) int64 block scale (1 unless 4 bits)
    q: np.ndarray       # (out, in) int64


def _fit_rows(W: np.ndarray, bits: int, imp: np.ndarray | None, block: int) -> _Fit:
    """Row (and for 4 bits block) scales and integer weights. imp: (in,) importance, or None for max-abs."""
    out, n_in = W.shape
    lo, hi = QRANGE[bits]
    nb = -(-n_in // block)
    cols = np.arange(n_in) // block
    absw = np.abs(W)
    if bits != 4:
        peak = absw.max(axis=1)
        base = np.where(peak > 0, peak / hi, 1.0)
        S = base.copy()
        if imp is not None:
            best = None
            for c in SCALE_GRID:
                s = base * c
                q = np.clip(_round(W / s[:, None]), lo, hi)
                err = ((W - q * s[:, None]) ** 2 * imp[None, :]).sum(axis=1)
                if best is None:
                    best, S = err, s.copy()
                else:
                    better = err < best
                    best = np.where(better, err, best)
                    S = np.where(better, s, S)
        q = np.clip(_round(W / S[:, None]), lo, hi).astype(np.int64)
        return _Fit(S, np.ones((out, nb), np.int64), q)
    # 4 bits: block real scale r = max|w| / 7, row scale S = max_b r / 63, integer d = round(r / S) in [1, 63].
    peak_b = np.zeros((out, nb))
    for b in range(nb):
        peak_b[:, b] = absw[:, b * block:(b + 1) * block].max(axis=1)
    r = peak_b / hi
    rmax = r.max(axis=1)
    S = np.where(rmax > 0, rmax / D_MAX, 1.0)
    d = np.clip(_round(r / S[:, None]), 1, D_MAX).astype(np.int64)
    if imp is not None:  # per block, the integer d (around the max-abs choice) with the least weighted error
        for b in range(nb):
            sl = slice(b * block, min((b + 1) * block, n_in))
            w, iw = W[:, sl], imp[sl][None, :]
            best = np.full(out, np.inf)
            choice = d[:, b].copy()
            for delta in range(-6, 3):
                cand = np.clip(d[:, b] + delta, 1, D_MAX)
                s = (S * cand)[:, None]
                q = np.clip(_round(w / s), lo, hi)
                err = ((w - q * s) ** 2 * iw).sum(axis=1)
                better = err < best
                best = np.where(better, err, best)
                choice = np.where(better, cand, choice)
            d[:, b] = choice
    scale = S[:, None] * d[:, cols]
    q = np.clip(_round(W / scale), lo, hi).astype(np.int64)
    return _Fit(S, d, q)


def quantize_layer(W: np.ndarray, b: np.ndarray, bits: int, first: bool, last: bool, x_max: np.ndarray,
                   imp: np.ndarray | None = None, block: int = BLOCK) -> dict:
    """One low-bit layer (see the module docstring). W: (out, in) float, already folded (first layer: / in_scale)."""
    W = np.asarray(W, np.float64)
    b = np.asarray(b, np.float64)
    s_in = 1.0 if first else 1.0 / ACT_ONE
    out, n_in = W.shape
    nb = -(-n_in // block)
    cols = np.arange(n_in) // block
    x_max = np.asarray(x_max, np.int64)
    fit = _fit_rows(W, bits, imp, block)
    for _ in range(40):  # widen the row scale until the integer bound holds (rare: large biases)
        B = _round(b / (fit.S * s_in))
        w_eff = fit.d[:, cols] * fit.q
        bound = np.abs(w_eff) @ x_max + np.abs(B)
        bad = bound > ACC_LIMIT
        if not bad.any():
            break
        fit.S = np.where(bad, fit.S * 2.0, fit.S)
        scale = fit.S[:, None] * fit.d[:, cols]
        fit.q = np.clip(_round(W / scale), *QRANGE[bits]).astype(np.int64)
    else:
        raise ValueError("layer cannot be bounded inside the int32 accumulator range")
    layer = {"bits": bits, "in": n_in, "out": out, "in_pad": nb * block, "block": block, "hidden": not last,
             "first": first, "q": fit.q.astype(np.int16 if bits == 16 else np.int8), "d": fit.d.astype(np.uint8),
             "b": B.astype(np.int32), "row_scale": fit.S.astype(np.float64), "bound": int(bound.max(initial=0)),
             "x_max": x_max.astype(np.int64)}
    if last:
        layer["out_scale"] = (fit.S * s_in).astype(np.float64)
    else:
        m = ACT_ONE * fit.S * s_in
        R = R_MAX
        while R > 0 and np.any(_round(m * 2.0 ** R) > 2 ** 31 - 1):
            R -= 1
        M = _round(m * 2.0 ** R).astype(np.int64)
        if np.any(M < 1):
            raise ValueError("requantization multiplier underflow (a zero row scale)")
        layer["mult"], layer["rshift"] = M.astype(np.int32), int(R)
    layer["w_err"] = float(((W - fit.S[:, None] * fit.d[:, cols] * fit.q) ** 2).sum())
    return layer


def quantize_mlp(mlp: MLP, in_scale: np.ndarray, in_lo: np.ndarray, in_hi: np.ndarray, head: str, profile: dict,
                 out_mult: float = 1.0, importance: list[np.ndarray] | None = None, block: int = BLOCK) -> dict:
    """A low-bit net for one policy head, with the same folding as quant.quantize (input scales, temperature)."""
    Ws = [w.astype(np.float64) for w in mlp.W]
    bs = [b.astype(np.float64) for b in mlp.b]
    Ws[-1], bs[-1] = Ws[-1] * out_mult, bs[-1] * out_mult
    Ws[0] = Ws[0] / np.asarray(in_scale, np.float64)[None, :]
    in_max = np.maximum(np.abs(np.asarray(in_lo)), np.abs(np.asarray(in_hi))).astype(np.int64)
    n = len(Ws)
    layers = []
    for i, (W, b) in enumerate(zip(Ws, bs)):
        x_max = in_max if i == 0 else np.full(W.shape[1], ACT_ONE, np.int64)
        imp = None if importance is None else np.asarray(importance[i], np.float64)
        layers.append(quantize_layer(W, b, layer_bits(profile, head, i, n), i == 0, i == n - 1, x_max, imp, block))
    return {"layers": layers, "in": int(Ws[0].shape[1]), "out": int(Ws[-1].shape[0]), "head": head}


# ----------------------------------------------------------------------------------------------- packing


def pack_int4(q: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """(out, in) int in [-8, 7] -> (out, nb * block / 2) uint8. In each block byte j holds q[j] + 8 in its low
    nibble and q[j + block/2] + 8 in its high nibble (padding columns are q = 0)."""
    out, n_in = q.shape
    nb = -(-n_in // block)
    pad = np.zeros((out, nb * block), np.int64)
    pad[:, :n_in] = q
    pad = (pad + 8).reshape(out, nb, 2, block // 2)
    return (pad[:, :, 0, :] | (pad[:, :, 1, :] << 4)).astype(np.uint8).reshape(out, nb * block // 2)


def unpack_int4(packed: np.ndarray, n_in: int, block: int = BLOCK) -> np.ndarray:
    out = packed.shape[0]
    nb = packed.shape[1] * 2 // block
    p = packed.reshape(out, nb, block // 2).astype(np.int64)
    q = np.stack([(p & 15) - 8, (p >> 4) - 8], axis=2).reshape(out, nb * block)
    return q[:, :n_in]


def packed_weights(layer: dict) -> np.ndarray:
    """The weight array a kernel reads: int16/int8 rows padded to in_pad, or packed int4."""
    q = layer["q"].astype(np.int64)
    if layer["bits"] == 4:
        return pack_int4(q, layer["block"])
    pad = np.zeros((layer["out"], layer["in_pad"]), np.int16 if layer["bits"] == 16 else np.int8)
    pad[:, :layer["in"]] = q
    return pad


def layer_arrays(layer: dict) -> dict[str, np.ndarray]:
    """Every array a deployed layer consists of (what the header and the digest contain)."""
    arr = {"w": packed_weights(layer), "b": layer["b"].astype(np.int32)}
    if layer["bits"] == 4:
        arr["d"] = layer["d"].astype(np.uint8)
    if layer["hidden"]:
        arr["mult"] = layer["mult"].astype(np.int32)
        arr["rshift"] = np.array(layer["rshift"], np.int32)
    else:
        arr["out_scale"] = layer["out_scale"].astype(np.float64)
    arr["meta"] = np.array([layer["bits"], layer["in"], layer["out"], layer["in_pad"], layer["block"],
                            int(layer["hidden"])], np.int32)
    return arr


def flatten(net: dict, prefix: str) -> dict[str, np.ndarray]:
    out = {}
    for i, layer in enumerate(net["layers"]):
        out |= {f"{prefix}.{i}.{k}": v for k, v in layer_arrays(layer).items()}
    return out


def unflatten(arrays: dict, prefix: str) -> dict:
    """Rebuild an emulation-ready net from flattened arrays (the inverse of flatten, independent of the fit)."""
    layers = []
    i = 0
    while f"{prefix}.{i}.w" in arrays:
        g = lambda k: arrays[f"{prefix}.{i}.{k}"]  # noqa: E731
        bits, n_in, n_out, in_pad, block, hidden = (int(v) for v in g("meta"))
        w = g("w")
        q = unpack_int4(w, n_in, block) if bits == 4 else w[:, :n_in].astype(np.int64)
        nb = in_pad // block
        layer = {"bits": bits, "in": n_in, "out": n_out, "in_pad": in_pad, "block": block, "hidden": bool(hidden),
                 "q": q, "d": g("d") if bits == 4 else np.ones((n_out, nb), np.uint8), "b": g("b")}
        if hidden:
            layer["mult"], layer["rshift"] = g("mult"), int(g("rshift"))
        else:
            layer["out_scale"] = g("out_scale")
        layers.append(layer)
        i += 1
    return {"layers": layers, "in": layers[0]["in"], "out": layers[-1]["out"]}


def digest(flat: dict) -> str:
    return quant.digest(flat)


# --------------------------------------------------------------------------------------------- emulation


def forward_acc(net: dict, x: np.ndarray, check: bool = True) -> np.ndarray:
    """Exact integer emulation: x (..., in) integers -> (..., out) int64 last-layer accumulators (bias included)."""
    lead = x.shape[:-1]
    h = np.asarray(x, np.int64).reshape(-1, x.shape[-1])
    for L in net["layers"]:
        cols = np.arange(L["in"]) // L["block"]
        w_eff = L["d"].astype(np.int64)[:, cols] * np.asarray(L["q"], np.int64)
        acc = h @ w_eff.T + L["b"].astype(np.int64)
        if check and np.abs(acc).max(initial=0) > ACC_LIMIT:
            raise OverflowError("accumulator outside the proven int32 range")
        if L["hidden"]:
            M = L["mult"].astype(np.int64)
            R = int(L["rshift"])
            v = acc * M + (np.int64(1) << (R - 1) if R > 0 else 0)
            h = np.clip(v >> R, 0, ACT_ONE)
        else:
            h = acc
    return h.reshape(*lead, h.shape[-1])


def to_real(net: dict, acc: np.ndarray) -> np.ndarray:
    return acc.astype(np.float64) * net["layers"][-1]["out_scale"]


# ------------------------------------------------------------------------------------------ whole policies


def quantize_policy(policy, layout, profile_name: str, profiles: dict | None = None,
                    importance: dict | None = None, block: int = BLOCK) -> dict:
    """All four heads of a PolicyNet under one profile. reference16 returns the legacy quant.quantize nets."""
    from training.export_weights import input_specs  # noqa: PLC0415 (training imports fusion, not the reverse at import time)

    profiles = profiles or PROFILES
    profile = profiles[profile_name]
    spec = input_specs(layout, getattr(policy, "family", "p2"))
    beta = policy.beta
    mult = {"mover": beta["joint"], "squad": beta["joint"], "act": beta["act"], "target": beta["target"]}
    if profile.get("legacy"):
        nets = {h: quant.quantize(getattr(policy, h), *spec[h], out_mult=mult[h]) for h in HEADS}
        return {"profile": profile_name, "legacy": True, "nets": nets, "family": policy.family,
                "identity": profile_identity(profile_name, profile, block)}
    nets = {h: quantize_mlp(getattr(policy, h), *spec[h], head=h, profile=profile, out_mult=mult[h],
                            importance=None if importance is None else importance.get(h), block=block) for h in HEADS}
    return {"profile": profile_name, "legacy": False, "nets": nets, "family": policy.family,
            "identity": profile_identity(profile_name, profile, block)}


def policy_flat(qp: dict) -> dict[str, np.ndarray]:
    flat = {}
    for h, net in qp["nets"].items():
        flat |= quant.flatten(net, h) if qp["legacy"] else flatten(net, h)
    return flat


def manifest_identity(qp: dict, sources: dict | None = None) -> dict:
    """The common identity of one exported profile: arrays, profile and format, feature family, sources."""
    flat = policy_flat(qp)
    return {"format": FORMAT_VERSION, "profile": qp["profile"], "profile_identity": qp["identity"],
            "family": qp["family"], "weights_digest": digest(flat), "sources": sources or {}}


def head_forward(qp: dict, head: str, x: np.ndarray) -> np.ndarray:
    """Real-valued outputs of one head under its profile (legacy path for reference16)."""
    net = qp["nets"][head]
    if qp["legacy"]:
        return quant.to_real(net, quant.forward_raw(net, x))
    return to_real(net, forward_acc(net, x))


def joint_logits(qp: dict, X) -> np.ndarray:
    """Masked-joint logits (B, 125) from a raw-integer PolicyBatch, combined in inference/policy.hpp's order."""
    from .models import combine_joint  # noqa: PLC0415

    s = head_forward(qp, "mover", X.mover_in)[..., 0]
    r = head_forward(qp, "squad", X.squad_in)
    return combine_joint(s, r)


# ------------------------------------------------------------------------------------------- accounting


def storage(net: dict, legacy: bool = False) -> dict:
    """Bytes of one head, measured from its arrays: packed weights, biases, scales, padding; scratch and copies."""
    rows = []
    for i, L in enumerate(net["layers"]):
        if legacy:
            w = L["w"]
            rows.append({"layer": i, "bits": 16, "in": w.shape[1], "out": w.shape[0], "weights": w.nbytes,
                         "bias": L["b"].nbytes, "scales": 0, "padding": 0, "params": int(w.size + L["b"].size)})
            continue
        arr = layer_arrays(L)
        pad_cols = L["in_pad"] - L["in"]
        rows.append({"layer": i, "bits": L["bits"], "in": L["in"], "out": L["out"], "weights": arr["w"].nbytes,
                     "bias": arr["b"].nbytes,
                     "scales": sum(arr[k].nbytes for k in ("d", "mult", "rshift", "out_scale") if k in arr),
                     "padding": pad_cols * L["out"] * L["bits"] // 8, "params": int(L["in"] * L["out"] + L["out"])})
    widest = max(max(r["in"], r["out"]) for r in rows)
    pad_widest = max(-(-r["in"] // BLOCK) * BLOCK for r in rows) if not legacy else widest
    scratch = 2 * 2 * max(widest, pad_widest) + 4 * widest  # two int16 activation buffers + one int32 sum buffer
    total = {k: sum(r[k] for r in rows) for k in ("weights", "bias", "scales", "padding", "params")}
    return {"layers": rows, **total, "resident_bytes": total["weights"] + total["bias"] + total["scales"],
            "scratch_bytes": scratch, "resident_copies": 1}


def policy_storage(qp: dict) -> dict:
    return {h: storage(net, qp["legacy"]) for h, net in qp["nets"].items()}
