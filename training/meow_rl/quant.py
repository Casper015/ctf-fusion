"""Integer CPU export of the actor in the repository's INT16 format (reference16, docs/P_MIXED_PRECISION.md).

The exported actor is exactly a fusion.quant net: int16 weights, int32 biases and accumulators proved inside
+-2^30, hidden activations on the fixed-point scale 0..128 (kActOne), power-of-two shifts, floor(x + 1/2)
rounding, real output = acc / 2^out_exp. The C++ bot evaluates it with inference/mlp.hpp (fusion::forwardRaw, the
SIMD kernels measured in reports/host/SIMD_OPTIMIZATION_REPORT.md); fusion.quant.forward_raw is its exact emulation.

Two steps fit the Meowmind actor (plain ReLU, unbounded) into that format without changing its function:
  1. inputs: the reference's features are floats clamped to [-4, 4]; the bot passes x_q = clamp(floor(x * 2^12
     + 1/2), -16384, 16384) and the first layer is quantized with in_scale 2^12 (fusion.quant's raw-integer inputs);
  2. activations: ReLU is positively homogeneous, so each hidden neuron o is divided by its calibrated range H_o
     (margin times its largest FP32 activation on recorded observations) and the next layer's column o multiplied
     by H_o. The float network is unchanged; the format's clipped ReLU [0, 1] then equals ReLU except where an
     activation exceeds H_o (it saturates; agreement() counts these).
The output scale 2^-out_exp is shared by every unit and move, so the greedy joint compares int64 sums of three
accumulators directly (ties to the lowest index); probabilities use logit / T exactly as the float actor.

Precision is the format's: hidden activations have 129 levels. On recorded games the warm-start actor's integer
greedy joint agrees with FP32 on about 97.5% of decisions (the T0054 measurement, docs/MEOWMIND_RL_IMPLEMENTATION.md);
agreement() reports it for every export. Collection always samples from the FP32 snapshot whose probabilities it
stores: the integer actor is a deployment artifact checked against it, never the behavior policy of training.
"""
from __future__ import annotations

import types

import numpy as np

from fusion import quant as FQ
from .identity import ACTOR_ARCH, POLICY_FEATURES, TEMPERATURE, digest
from .policy import JA, JB, JC, check_actor, joint_log_probs

IN_BITS = 12
IN_MAX = 4 << IN_BITS                    # features are clamped to [-4, 4]
SCHEME = "meow_rl_reference16_v1"
MARGIN = 1.1
FLOOR = 1e-3                             # range of a neuron never active in calibration


def quantize_input(feats) -> np.ndarray:
    x = np.asarray(feats, np.float32).astype(np.float64)
    return np.clip(np.floor(x * (1 << IN_BITS) + 0.5), -IN_MAX, IN_MAX).astype(np.int64)


RANGE_BITS = 8                           # calibrated ranges are rounded up to 8 significant bits (range_of)


def activations(w: dict, feats) -> list[np.ndarray]:
    """Hidden activations of feats [..., 112] in float64 (the calibration statistics)."""
    x = np.asarray(feats, np.float32).astype(np.float64).reshape(-1, w["w0"].shape[1])
    h0 = np.maximum(0.0, x @ w["w0"].astype(np.float64).T + w["b0"].astype(np.float64))
    return [h0, np.maximum(0.0, h0 @ w["w1"].astype(np.float64).T + w["b1"].astype(np.float64))]


def range_of(peak: np.ndarray, margin: float) -> np.ndarray:
    """A neuron's range: margin x its calibrated peak, at least FLOOR, rounded up to RANGE_BITS significant bits.
    The rounding makes the export independent of the last bits of the statistics (BLAS summation order, compiler FMA
    in the features), so the same inputs give the same integers on every platform."""
    r = np.maximum(np.asarray(peak, np.float64) * margin, FLOOR)
    step = np.exp2(np.floor(np.log2(r)) - RANGE_BITS)
    return np.ceil(r / step) * step


def fold(w: dict, ranges: list[np.ndarray]) -> types.SimpleNamespace:
    """The same function with every hidden neuron rescaled to [0, 1] over its range (an MLP for fusion.quant)."""
    H0, H1 = (np.asarray(r, np.float64) for r in ranges)
    W = [w["w0"] / H0[:, None], w["w1"] * H0[None, :] / H1[:, None], w["w2"] * H1[None, :]]
    b = [w["b0"] / H0, w["b1"] / H1, w["b2"].astype(np.float64)]
    return types.SimpleNamespace(W=[x.astype(np.float64) for x in W], b=[x.astype(np.float64) for x in b])


def quantize(actor: dict, calibration: np.ndarray, margin: float = MARGIN) -> dict:
    """The INT16 actor, calibrated on feats [..., 112] from recorded observations (required)."""
    w = check_actor(actor)
    calibration = np.asarray(calibration, np.float32)
    if calibration.size == 0:
        raise ValueError("calibration observations are required")
    ranges = [range_of(a.max(axis=0), margin) for a in activations(w, calibration)]
    net = FQ.quantize(fold(w, ranges), np.full(w["w0"].shape[1], float(1 << IN_BITS)),
                      np.full(w["w0"].shape[1], -IN_MAX), np.full(w["w0"].shape[1], IN_MAX))
    FQ.check_widths(net, "meow_rl actor")
    flat = FQ.flatten(net, "actor")
    return {"scheme": SCHEME, "format": "reference16", "actor": ACTOR_ARCH, "policy_features": POLICY_FEATURES,
            "temperature": TEMPERATURE, "in_bits": IN_BITS, "net": net, "digest": FQ.digest(flat),
            "calibration": {"observations": int(calibration.size // w["w0"].shape[1]), "margin": margin,
                            "ranges": [r.tolist() for r in ranges]}}


def forward_int(q: dict, feats) -> np.ndarray:
    """Exact int32-range output accumulators (as int64) for feats [..., 112] -> [...]."""
    return FQ.forward_raw(q["net"], quantize_input(feats))[..., 0]


def saturated(q: dict, feats) -> list[int]:
    """Hidden activations clamped at kActOne (real activation above its calibrated range), per layer."""
    h = quantize_input(feats).reshape(-1, q["net"]["in"])
    out = []
    for layer in q["net"]["layers"]:
        acc = h @ layer["w"].astype(np.int64).T + layer["b"].astype(np.int64)
        if not layer["hidden"]:
            break
        s = layer["shift"]
        r = (acc + (1 << (s - 1))) >> s if s > 0 else acc
        out.append(int((r > FQ.ACT_ONE).sum()))
        h = np.clip(r, 0, FQ.ACT_ONE)
    return out


def joint_acc(q: dict, feats: np.ndarray) -> np.ndarray:
    """int64 joint accumulators [..., 125] from feats [..., 3, 5, 112] (unmasked)."""
    u = forward_int(q, feats)
    return u[..., 0, JA] + u[..., 1, JB] + u[..., 2, JC]


def greedy(q: dict, feats: np.ndarray, mask: np.ndarray) -> int:
    z = np.where(mask, joint_acc(q, feats), np.iinfo(np.int64).min)
    return int(np.argmax(z))


def log_probs(q: dict, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.ldexp(joint_acc(q, feats).astype(np.float64), -q["net"]["out_exp"]) / TEMPERATURE
    z = np.where(mask, z, -np.inf)
    m = z.max(axis=-1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=-1, keepdims=True))


def agreement(actor: dict, q: dict, feats: np.ndarray, masks: np.ndarray) -> dict:
    """Greedy-joint agreement, log-probability error and saturations of the integer actor against FP32."""
    lp32 = joint_log_probs(check_actor(actor), feats, masks).astype(np.float64)
    lpq = log_probs(q, feats, masks)
    same = np.mean([greedy(q, f, m) == int(np.argmax(l)) for f, m, l in zip(feats, masks, lp32)])
    ok = np.isfinite(lp32)
    err = np.abs(lp32[ok] - lpq[ok])
    return {"observations": len(feats), "greedy_agreement": float(same), "max_abs_logp_error": float(err.max()),
            "mean_abs_logp_error": float(err.mean()),
            "max_abs_prob_error": float(np.abs(np.exp(lp32[ok]) - np.exp(lpq[ok])).max()),
            "saturated": saturated(q, np.asarray(feats).reshape(-1, q["net"]["in"]))}


def header(q: dict, name: str, identity: dict) -> str:
    """A C++ header declaring the actor as a fusion::Net (inference/mlp.hpp), namespace meow_rl_model."""
    from training.export_weights import _array, _emit_nets  # noqa: PLC0415
    body: list[str] = []
    _emit_nets(body, {"actor": q["net"]}, {}, False, _array)
    return "\n".join([f"// Generated by training.meow_rl.quant.header: {name}. Do not edit.",
                      f"// scheme {q['scheme']} (reference16 INT16), actor {q['actor']}, features {q['policy_features']}",
                      f"// calibration {q['calibration']['observations']} observations, margin {q['calibration']['margin']}",
                      "#pragma once", "#include <cstdint>", '#include "mlp.hpp"', "namespace meow_rl_model {",
                      f'inline constexpr const char* kName = "{name}";',
                      f'inline constexpr const char* kScheme = "{q["scheme"]}";',
                      f'inline constexpr const char* kDigest = "{q["digest"]}";',
                      f'inline constexpr const char* kIdentity = "{digest(identity)}";',
                      f"inline constexpr int kInBits = {q['in_bits']};",
                      f"inline constexpr double kTemperature = {q['temperature']!r};", *body,
                      f"inline constexpr fusion::Net kActor{{actor_layers, {len(q['net']['layers'])}, {q['net']['out_exp']}}};",
                      "}  // namespace meow_rl_model", ""])
