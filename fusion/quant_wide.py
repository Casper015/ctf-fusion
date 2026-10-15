"""WideAct16: an integer MLP format with 15-bit hidden activations and INT64 accumulators (task T0061).

The higher-precision sibling of reference16 (fusion/quant.py, inference/mlp.hpp). Both keep int16 weights, power-of-two
scales and the same layer structure, but reference16 clips hidden activations to 0..128 (129 levels, ~7 bits) and keeps
every accumulator inside +-2^30 for int32 arithmetic. WideAct16 keeps hidden activations on 0..32767 (2^15 = 1.0,
32,768 levels, int16 storage) and accumulates in int64. It is evaluated by inference/mlp_wide.hpp; forward_raw below is
its exact emulation.

    layer 1 (raw integer inputs x, |x| <= in_max):
        Wq = round(W / in_scale * 2^a),  Bq = round(b * 2^a)                       acc ~ 2^a z
    layer l > 1 (inputs h ~ 2^15 h_real):
        Wq = round(W * 2^a),             Bq = round(b * 2^(a + 15))                acc ~ 2^(a+15) z
    hidden:  h' = act(acc, shift), shift = a - 15 (layer 1) or a (later)          h' ~ 2^15 relu(z), clamped to 32767
    output:  real = acc / 2^out_exp,  out_exp = a (layer 1) or a + 15 (later)

    act(acc, s) = 0                                   if acc <= 0
                = min((acc + 2^(s-1)) >> s, 32767)    if acc > 0 and s > 0   (round half up of acc / 2^s)
                = min(acc, 32767)                     if acc > 0 and s == 0
The right shift is applied only to positive values, so it is the same on every compiler (no implementation-defined
shift of a negative number). For acc <= 0 every rounding of acc / 2^s is <= 0, so act equals
clamp(floor(acc / 2^s + 1/2), 0, 32767) everywhere: the format's rounding is defined for negative values too.

Proved bounds (quantize refuses a layer that violates one):
  * weights: |Wq| <= 32767 (never -32768);
  * accumulator: sum_i |Wq[o, i]| * x_max[i] + |Bq[o]| <= 2^52 for every output o, where x_max is the feature spec's
    bound (layer 1) or 32767 (later). Every partial sum is bounded by the same total, so int64 addition in any order is
    exact, and the final accumulator converts to a double exactly;
  * SIMD pair sums: 2 * 32767 * max(x_max) < 2^31, so a pairwise int16 multiply-add into int32 (pmaddwd) cannot
    overflow before it is widened to int64 (inference/mlp_wide.hpp);
  * the bias fits int64 (it is inside the accumulator bound).
The exponent a is the largest that keeps |Wq| <= 32767 (and at most A_MAX); the accumulator bound never binds at these
widths (it is checked anyway). Layer 1 needs a >= 15 (shift >= 0), as reference16 needs a >= 7.
"""
from __future__ import annotations

import hashlib

import numpy as np

SCHEME = "wideact16_v1"
ACT_BITS = 15
ACT_ONE = 1 << ACT_BITS           # the real value 1.0
ACT_MAX = ACT_ONE - 1             # largest stored activation (int16)
W_LIMIT = 32767
ACC_LIMIT = 1 << 52
PAIR_LIMIT = (1 << 31) - 1
A_MAX = 40


def _round(x) -> np.ndarray:
    return np.floor(np.asarray(x, np.float64) + 0.5).astype(np.int64)


def _bound(Wq: np.ndarray, x_max: np.ndarray, Bq: np.ndarray) -> int:
    """max_o sum_i |Wq[o,i]| x_max[i] + |Bq[o]|, exactly. The int64 arithmetic of the proof itself cannot wrap: the
    weight sum is at most in * 32767 * max(x_max) < 2^63 (checked), and the bias term is added in Python integers."""
    if Wq.shape[1] * W_LIMIT * int(x_max.max()) >= 1 << 62:
        raise ValueError("layer too wide for the int64 bound computation")
    w_sum = np.abs(Wq).astype(np.int64) @ x_max.astype(np.int64)
    return max(int(s) + abs(int(c)) for s, c in zip(w_sum, Bq))


def quantize(W: list, b: list, in_scale, in_lo, in_hi, a_max: int = A_MAX) -> dict:
    """Quantize a float MLP (ReLU hidden layers, already folded so hidden activations live in [0, 1]) that reads
    inputs x / in_scale, for raw integer inputs bounded by in_lo..in_hi."""
    Ws = [np.asarray(w, np.float64) for w in W]
    bs = [np.asarray(c, np.float64) for c in b]
    Ws[0] = Ws[0] / np.asarray(in_scale, np.float64)[None, :]
    in_max = np.maximum(np.abs(np.asarray(in_lo, np.int64)), np.abs(np.asarray(in_hi, np.int64)))
    if int(in_max.max()) > ACT_MAX:                     # the native layer input is int16: +-32767 on both sides
        raise ValueError("inputs outside +-32767 are not representable in the int16 layer input")
    if 2 * W_LIMIT * int(max(int(in_max.max()), ACT_MAX)) > PAIR_LIMIT:
        raise ValueError("inputs too large for exact int16 pair sums")
    layers = []
    for i, (w, c) in enumerate(zip(Ws, bs)):
        first, last = i == 0, i == len(Ws) - 1
        peak = max(float(np.abs(w).max()), 1e-300)
        a = min(int(np.floor(np.log2(W_LIMIT / peak))), a_max)
        x_max = in_max if first else np.full(w.shape[1], ACT_MAX, np.int64)
        while True:
            # a float estimate first, so that rounding never sees a value outside int64; the exact proof follows
            est = float(((np.abs(w) * 2.0 ** a) @ x_max.astype(np.float64) + np.abs(c) * 2.0 ** (a if first else a + ACT_BITS)).max())
            if est > ACC_LIMIT:
                a -= 1
                continue
            Wq = _round(w * 2.0 ** a)
            Bq = _round(c * 2.0 ** (a if first else a + ACT_BITS))
            bound = _bound(Wq, x_max, Bq)
            if int(np.abs(Wq).max()) <= W_LIMIT and bound <= ACC_LIMIT:
                break
            a -= 1
        if first and not last and a < ACT_BITS:
            raise ValueError(f"layer 1 weights too large for WideAct16 (exponent {a} < {ACT_BITS})")
        if not first and not last and a < 0:
            raise ValueError(f"layer {i + 1} weights too large for WideAct16 (exponent {a} < 0)")
        layers.append({"w": Wq.astype(np.int16), "b": Bq.astype(np.int64), "shift": int(a - ACT_BITS if first else a),
                       "hidden": not last, "exp": int(a if first else a + ACT_BITS), "bound": int(bound)})
    return {"scheme": SCHEME, "layers": layers, "out_exp": layers[-1]["exp"], "in": int(Ws[0].shape[1]),
            "out": int(Ws[-1].shape[0]), "in_max": int(in_max.max())}


def activate(acc: np.ndarray, shift: int) -> np.ndarray:
    acc = np.asarray(acc, np.int64)
    pos = np.maximum(acc, 0)
    r = (pos + (1 << (shift - 1))) >> shift if shift > 0 else pos
    return np.minimum(r, ACT_MAX)


def forward_raw(q: dict, x: np.ndarray, check: bool = True) -> np.ndarray:
    """Exact emulation of fusion::wide::forwardRaw: integer inputs (..., in) -> int64 output accumulators (..., out)."""
    lead = x.shape[:-1]
    h = np.asarray(x, np.int64).reshape(-1, x.shape[-1])
    if check and np.abs(h).max(initial=0) > q["in_max"]:
        raise OverflowError("input outside the proven range")
    for layer in q["layers"]:
        acc = h @ layer["w"].astype(np.int64).T + layer["b"].astype(np.int64)
        if check and np.abs(acc).max(initial=0) > ACC_LIMIT:
            raise OverflowError("accumulator outside the proven range")
        h = activate(acc, layer["shift"]) if layer["hidden"] else acc
    return h.reshape(*lead, h.shape[-1])


def to_real(q: dict, acc: np.ndarray) -> np.ndarray:
    return np.ldexp(np.asarray(acc, np.int64).astype(np.float64), -q["out_exp"])


def saturated(q: dict, x: np.ndarray) -> list[int]:
    """Per hidden layer, how many activations were clamped at ACT_MAX (a real activation at or above its range)."""
    h = np.asarray(x, np.int64).reshape(-1, q["in"])
    out = []
    for layer in q["layers"]:
        if not layer["hidden"]:
            break
        acc = h @ layer["w"].astype(np.int64).T + layer["b"].astype(np.int64)
        s = layer["shift"]
        r = (np.maximum(acc, 0) + (1 << (s - 1))) >> s if s > 0 else np.maximum(acc, 0)
        out.append(int((r > ACT_MAX).sum()))
        h = np.minimum(r, ACT_MAX)
    return out


def flatten(q: dict, prefix: str) -> dict:
    out = {f"{prefix}.out_exp": np.array(q["out_exp"]), f"{prefix}.scheme": np.array(SCHEME)}
    for i, layer in enumerate(q["layers"]):
        for k in ("w", "b", "shift", "hidden"):
            out[f"{prefix}.{i}.{k}"] = np.asarray(layer[k])
    return out


def digest(flat: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(flat):
        v = np.ascontiguousarray(flat[k])
        h.update(f"{k}|{v.dtype.str}|{v.shape}|".encode())
        h.update(v.tobytes())
    return h.hexdigest()


def describe(q: dict) -> dict:
    """Widths, scales, bounds and sizes, for the export manifest."""
    layers = q["layers"]
    return {"scheme": SCHEME, "weights": "int16", "biases": "int64", "accumulator": "int64",
            "hidden_activation": f"int16 storage, 0..{ACT_MAX} (2^{ACT_BITS} = 1.0, {ACT_ONE} levels)",
            "rounding": "round half up of acc / 2^shift on positive accumulators; acc <= 0 -> 0",
            "acc_limit": ACC_LIMIT, "pair_limit": PAIR_LIMIT,
            "layers": [{"in": int(L["w"].shape[1]), "out": int(L["w"].shape[0]), "exp": L["exp"], "shift": L["shift"],
                        "hidden": L["hidden"], "bound": L["bound"], "bound_log2": float(np.log2(max(L["bound"], 1))),
                        "max_abs_weight": int(np.abs(L["w"]).max())}
                       for L in layers],
            "parameter_bytes": int(sum(L["w"].size * 2 + L["b"].size * 8 for L in layers)),
            "peak_buffer_bytes": int(max(max(L["w"].shape) for L in layers) * (2 * 2 + 8))}
