"""Integer export format and its exact emulation of inference/mlp.hpp.

A quantized net is a list of layers {w int16 [out][in], b int32, shift, hidden}
plus `out_exp`. With raw integer inputs x (the extractor's features):

    layer 1:  acc = W1q x + B1q,  W1q = round(W1 / scale * 2^a1)          acc ~ 2^a1 z
              h = clamp(round(acc / 2^(a1-7)), 0, 128)                      h ~ 128 relu1(z)
    layer l:  acc = Wq h + Bq,    Wq = round(W * 2^al), Bq = round(b * 2^(al+7))
              hidden: h = clamp(round(acc / 2^al), 0, 128)
              output: real = acc / 2^(al+7)

`a` is the largest exponent that keeps every weight in int16 and every
accumulator inside +-2^30 for any input allowed by the feature spec, so the
int32 C++ arithmetic can never overflow; `quantize` proves this bound.
"""
from __future__ import annotations

import numpy as np

from .nn import MLP

ACT_BITS = 7                 # kActOne = 128
ACT_ONE = 1 << ACT_BITS
ACC_LIMIT = 1 << 30
W_LIMIT = 32767


def _round(x):
    return np.floor(np.asarray(x, np.float64) + 0.5).astype(np.int64)


def quantize(mlp: MLP, in_scale: np.ndarray, in_lo: np.ndarray, in_hi: np.ndarray,
             out_rows=None, out_mult: float = 1.0) -> dict:
    """Quantize a float MLP that was trained on inputs x / in_scale.

    out_rows keeps a subset of output rows; out_mult multiplies the output
    (a fitted inverse temperature).
    """
    Ws = [w.astype(np.float64) for w in mlp.W]
    bs = [b.astype(np.float64) for b in mlp.b]
    if out_rows is not None:
        Ws[-1], bs[-1] = Ws[-1][list(out_rows)], bs[-1][list(out_rows)]
    Ws[-1], bs[-1] = Ws[-1] * out_mult, bs[-1] * out_mult
    Ws[0] = Ws[0] / np.asarray(in_scale, np.float64)[None, :]
    in_max = np.maximum(np.abs(in_lo), np.abs(in_hi)).astype(np.int64)
    layers = []
    for i, (W, b) in enumerate(zip(Ws, bs)):
        first, last = i == 0, i == len(Ws) - 1
        peak = max(float(np.abs(W).max()), 1e-12)
        a = int(np.floor(np.log2(W_LIMIT / peak)))
        a = min(a, 24)
        x_max = in_max if first else np.full(W.shape[1], ACT_ONE, np.int64)
        while True:
            Wq = _round(W * 2.0 ** a)
            Bq = _round(b * 2.0 ** (a if first else a + ACT_BITS))
            bound = (np.abs(Wq) @ x_max + np.abs(Bq)).max()
            if np.abs(Wq).max() <= W_LIMIT and bound <= ACC_LIMIT:
                break
            a -= 1
        if first and not last and a < ACT_BITS:
            raise ValueError(f"layer 1 weights too large to quantize (exponent {a})")
        shift = (a - ACT_BITS) if first else a
        layers.append({"w": Wq.astype(np.int16), "b": Bq.astype(np.int32), "shift": int(shift), "hidden": not last,
                       "exp": int(a if first else a + ACT_BITS), "bound": int(bound)})
    return {"layers": layers, "out_exp": layers[-1]["exp"], "in": int(Ws[0].shape[1]), "out": int(Ws[-1].shape[0])}


def _round_shift(v: np.ndarray, s: int) -> np.ndarray:
    return (v + (1 << (s - 1))) >> s if s > 0 else v


def forward_raw(q: dict, x: np.ndarray) -> np.ndarray:
    """Exact emulation of fusion::forwardRaw; x: (..., in) integers -> (..., out) int64 accumulators."""
    lead = x.shape[:-1]
    h = np.asarray(x, np.int64).reshape(-1, x.shape[-1])
    for layer in q["layers"]:
        acc = h @ layer["w"].astype(np.int64).T + layer["b"].astype(np.int64)
        if np.abs(acc).max(initial=0) > ACC_LIMIT:
            raise OverflowError("accumulator outside the proven int32 range")
        h = np.clip(_round_shift(acc, layer["shift"]), 0, ACT_ONE) if layer["hidden"] else acc
    return h.reshape(*lead, h.shape[-1])


def to_real(q: dict, acc: np.ndarray) -> np.ndarray:
    return np.ldexp(acc.astype(np.float64), -q["out_exp"])


def flatten(q: dict, prefix: str) -> dict:
    out = {f"{prefix}.out_exp": np.array(q["out_exp"])}
    for i, layer in enumerate(q["layers"]):
        for k in ("w", "b", "shift", "hidden"):
            out[f"{prefix}.{i}.{k}"] = np.asarray(layer[k])
    return out


def unflatten(arrays: dict, prefix: str) -> dict:
    layers = []
    i = 0
    while f"{prefix}.{i}.w" in arrays:
        layers.append({k: arrays[f"{prefix}.{i}.{k}"] for k in ("w", "b")} |
                      {"shift": int(arrays[f"{prefix}.{i}.shift"]), "hidden": bool(arrays[f"{prefix}.{i}.hidden"])})
        i += 1
    return {"layers": layers, "out_exp": int(arrays[f"{prefix}.out_exp"]),
            "in": layers[0]["w"].shape[1], "out": layers[-1]["w"].shape[0]}
