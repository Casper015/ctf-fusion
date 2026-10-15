"""The three CPU critic backends of the search diagnosis (task T0061, docs/MEOWMIND_RL_ALIGNED.md).

One trained critic (416 public value features -> hidden ReLU layers -> one linear value in 2 U - 1 units), three
numerical formats, each a C++ header in namespace meow_rl_critic with the same interface:

    kFormat, kDigest, kIdentity
    double value(const float* f)          // f: the 416 value features of learning::Geometry::value

  * fp32       (arm F): the checkpoint's float32 weights unchanged, float32 arithmetic (inference/mlp_float.hpp);
  * reference16 (arm R): the repository's INT16 format, hidden activations 0..128, int32 accumulators (fusion/quant.py,
    inference/mlp.hpp), exactly the T0056 critic export (export.quantize_critic);
  * wideact16  (arm H): int16 weights, hidden activations 0..32767, int64 accumulators (fusion/quant_wide.py,
    inference/mlp_wide.hpp).
R and H fold the same per-neuron ranges from the same calibration corpus (the aligned critic scale rule, SCALE_RULE:
training.meow_rl.quant.range_of, i.e. margin x the calibrated FP32 peak rounded up to 8 significant bits, except that a
neuron never active in calibration gets the median range of its layer's active neurons instead of FLOOR), and both
quantize the inputs as clamp(floor(x * 2^12 + 1/2), -16384, 16384). They differ only in what the formats themselves
differ in: activation resolution, accumulator width and therefore the weight exponent the bound admits.

Why the dead-neuron rule: every layer has one power-of-two weight exponent. Folding a never-active neuron with the
range FLOOR (1e-3) multiplies its weights by ~1000, and the layer's exponent then follows that one row: every other
neuron loses ~10 bits. reference16 rarely notices (its exponent is usually set by the int32 accumulator bound), but a
format whose exponent follows the weight peak does. "reference16_t0056" (diagnostic only, never deployed) is the
T0056/T0057 critic export unchanged (export.quantize_critic), kept to relate R to the critic T0057 screened.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from fusion import quant as FQ  # noqa: E402
from fusion import quant_wide as FW  # noqa: E402
from training.meow_rl import identity as I  # noqa: E402
from training.meow_rl import quant as Q  # noqa: E402

FORMATS = ("fp32", "reference16", "wideact16")
DIAGNOSTIC_FORMATS = ("reference16_t0056",)
SCALE_RULE = "aligned_critic_scale_v1: range_of(margin x peak), never-active neurons -> layer median active range"
WIDTH = 1024                                  # widest critic layer (large_v: 416 -> 1024 -> 1024 -> 512 -> 1)


def _export_module():
    spec = importlib.util.spec_from_file_location("meow_rl_export", REPO / "bots" / "meow_rl" / "export.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


E = _export_module()


def ranges(W, b, calibration: np.ndarray, margin: float = Q.MARGIN) -> list[np.ndarray]:
    """Per hidden layer, each neuron's range under SCALE_RULE (and how many neurons were never active)."""
    out = []
    for a in E.critic_forward(W, b, calibration, hidden=True):
        peak = a.max(axis=0)
        r = Q.range_of(peak, margin)
        live = peak > 0
        if live.any() and not live.all():
            r = np.where(live, r, np.median(r[live]))
        out.append(r)
    return out


def dead_neurons(W, b, calibration: np.ndarray) -> list[int]:
    return [int((a.max(axis=0) <= 0).sum()) for a in E.critic_forward(W, b, calibration, hidden=True)]


def fold(W, b, rng: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The same function with every hidden neuron divided by its range (ReLU is positively homogeneous)."""
    Wf, bf, prev = [], [], None
    for i, (w, c) in enumerate(zip(W, b)):
        w, c = np.asarray(w, np.float64), np.asarray(c, np.float64)
        if prev is not None:
            w = w * prev[None, :]
        if i < len(W) - 1:
            w, c = w / rng[i][:, None], c / rng[i]
            prev = rng[i]
        Wf.append(w), bf.append(c)
    return Wf, bf


def build(fmt: str, W, b, calibration: np.ndarray, margin: float = Q.MARGIN) -> dict:
    """One backend of critic (W, b): {"format", "digest", ...}; R and H need the calibration value features."""
    W = [np.asarray(w, np.float32) for w in W]
    b = [np.asarray(c, np.float32) for c in b]
    if fmt == "fp32":
        h = hashlib.sha256()
        for i, (w, c) in enumerate(zip(W, b)):
            h.update(f"{i}|{w.shape}|".encode()), h.update(w.tobytes()), h.update(c.tobytes())
        return {"format": "fp32", "W": W, "b": b, "digest": h.hexdigest(),
                "describe": {"weights": "float32", "arithmetic": "float32, 8 interleaved partial sums per row",
                             "parameter_bytes": int(sum(w.size + c.size for w, c in zip(W, b)) * 4)}}
    calibration = np.asarray(calibration, np.float32)
    if calibration.size == 0:
        raise ValueError("an integer critic needs calibration observations")
    rng = ranges(W, b, calibration, margin)
    n_in = W[0].shape[1]
    cal = {"margin": margin, "observations": int(len(calibration)), "range_bits": Q.RANGE_BITS}
    cal["dead_neurons"] = dead_neurons(W, b, calibration)
    if fmt in ("reference16", "reference16_t0056"):
        if fmt == "reference16_t0056":
            q = E.quantize_critic(W, b, calibration, margin)
            net, dig = q["net"], q["digest"]
        else:
            cal["scale_rule"] = SCALE_RULE
            Wf, bf = fold(W, b, rng)
            net = FQ.quantize(types.SimpleNamespace(W=Wf, b=bf), np.full(n_in, float(1 << Q.IN_BITS)),
                              np.full(n_in, -Q.IN_MAX), np.full(n_in, Q.IN_MAX))
            FQ.check_widths(net, "meow_rl critic", WIDTH)
            dig = FQ.digest(FQ.flatten(net, "critic"))
        return {"format": fmt, "net": net, "digest": dig, "calibration": cal,
                "describe": {"weights": "int16", "biases": "int32", "accumulator": "int32 (proved inside +-2^30)",
                             "hidden_activation": "0..128 (129 levels)",
                             "layers": [{"exp": L["exp"], "shift": L["shift"], "bound": L["bound"],
                                         "max_abs_weight": int(np.abs(L["w"]).max())} for L in net["layers"]],
                             "parameter_bytes": int(sum(L["w"].size * 2 + L["b"].size * 4 for L in net["layers"]))}}
    if fmt == "wideact16":
        cal["scale_rule"] = SCALE_RULE
        Wf, bf = fold(W, b, rng)
        q = FW.quantize(Wf, bf, np.full(n_in, float(1 << Q.IN_BITS)), np.full(n_in, -Q.IN_MAX), np.full(n_in, Q.IN_MAX))
        return {"format": "wideact16", "net": q, "digest": FW.digest(FW.flatten(q, "critic")), "calibration": cal,
                "describe": FW.describe(q)}
    raise ValueError(f"unknown critic format {fmt!r}: {FORMATS + DIAGNOSTIC_FORMATS}")


def fp32_forward(W, b, x: np.ndarray, chunk: int = 256) -> np.ndarray:
    """Exact emulation of fusion::flt::forward (inference/mlp_float.hpp): float32 operations in its order, [N] values."""
    x = np.asarray(x, np.float32)
    out = []
    for start in range(0, len(x), chunk):
        h = x[start:start + chunk]
        for i, (w, c) in enumerate(zip(W, b)):
            w, c = np.asarray(w, np.float32), np.asarray(c, np.float32)
            n_out, n_in = w.shape
            P = np.zeros((len(h), n_out, 8), np.float32)
            full = n_in // 8 * 8
            for s in range(0, full, 8):
                P += w[None, :, s:s + 8] * h[:, None, s:s + 8]
            for k, s in enumerate(range(full, n_in)):
                P[:, :, k] += w[None, :, s] * h[:, None, s]
            comb = ((P[..., 0] + P[..., 4]) + (P[..., 2] + P[..., 6])) + ((P[..., 1] + P[..., 5]) + (P[..., 3] + P[..., 7]))
            y = c[None, :] + comb
            h = np.where(y > 0, y, np.float32(0)) if i < len(W) - 1 else y
        out.append(h[:, 0])
    return np.concatenate(out).astype(np.float64) if out else np.zeros(0)


def value(backend: dict, value_feats: np.ndarray) -> np.ndarray:
    """The backend's value of each row of value_feats [N, 416] (float64), exactly as its native header computes it."""
    x = np.asarray(value_feats, np.float32)
    if backend["format"] == "fp32":
        return fp32_forward(backend["W"], backend["b"], x)
    xi = Q.quantize_input(x)
    if backend["format"].startswith("reference16"):
        return FQ.to_real(backend["net"], FQ.forward_raw(backend["net"], xi))[:, 0]
    return FW.to_real(backend["net"], FW.forward_raw(backend["net"], xi))[:, 0]


def saturation(backend: dict, value_feats: np.ndarray) -> list[int] | None:
    """Hidden activations clamped at the top of their range, per layer (None for fp32)."""
    if backend["format"] == "fp32":
        return None
    xi = Q.quantize_input(np.asarray(value_feats, np.float32))
    net = backend["net"]
    if backend["format"] == "wideact16":
        return FW.saturated(net, xi)
    h, out = xi.reshape(-1, net["in"]), []
    for layer in net["layers"]:
        if not layer["hidden"]:
            break
        acc = h @ layer["w"].astype(np.int64).T + layer["b"].astype(np.int64)
        s = layer["shift"]
        r = (acc + (1 << (s - 1))) >> s if s > 0 else acc
        out.append(int((r > FQ.ACT_ONE).sum()))
        h = np.clip(r, 0, FQ.ACT_ONE)
    return out


# ---- C++ headers ------------------------------------------------------------------------------------------------

def _ints(name: str, ctype: str, values) -> str:
    flat = [str(int(v)) for v in np.asarray(values).reshape(-1)]
    lines = [", ".join(flat[i:i + 24]) for i in range(0, len(flat), 24)]
    return f"inline constexpr {ctype} {name}[] = {{\n    " + ",\n    ".join(lines) + "\n};"


def _float_literal(v: np.float32) -> str:
    s = "%.9g" % float(v)                       # 9 significant digits round-trip every float32 exactly
    if not any(ch in s for ch in ".enN"):
        s += "."
    return s + "f"


def _floats(name: str, values) -> str:
    flat = [_float_literal(v) for v in np.asarray(values, np.float32).reshape(-1)]
    lines = [", ".join(flat[i:i + 16]) for i in range(0, len(flat), 16)]
    return f"alignas(32) inline constexpr float {name}[] = {{\n    " + ",\n    ".join(lines) + "\n};"


INPUT = """inline int32_t quantize(float x) {          // training.meow_rl.quant.quantize_input: floor(x * 2^12 + 1/2), +-16384
    double v = std::floor(double(x) * 4096.0 + 0.5);
    return int32_t(v < -16384.0 ? -16384.0 : v > 16384.0 ? 16384.0 : v);
}"""


def header(backend: dict, name: str, identity: dict) -> str:
    fmt = backend["format"]
    top = [f"// Generated by bots/meow_rl/critics.py: {name} critic, format {fmt}. Do not edit.",
           f"// critic {identity['critic']}, units {identity['reward_units']} (docs/MEOWMIND_RL_ALIGNED.md)",
           "#pragma once", "#include <cmath>", "#include <cstdint>"]
    body = []
    if fmt == "fp32":
        top.append('#include "mlp_float.hpp"')
        rows = []
        for i, (w, c) in enumerate(zip(backend["W"], backend["b"])):
            body += [_floats(f"critic_w{i}", w), _floats(f"critic_b{i}", c)]
            rows.append(f"    {{{w.shape[1]}, {w.shape[0]}, critic_w{i}, critic_b{i}, {'true' if i < len(backend['W']) - 1 else 'false'}}},")
            body.append(f'static_assert({max(w.shape)} <= fusion::flt::kMaxWidth, "critic layer {i} exceeds the float buffers");')
        body += ["inline constexpr fusion::flt::Layer critic_layers[] = {", *rows, "};",
                 f"inline constexpr fusion::flt::Net kCritic{{critic_layers, {len(rows)}}};",
                 "inline double value(const float* f) {", "    float out[1];", "    fusion::flt::forward(kCritic, f, out);",
                 "    return double(out[0]);", "}"]
    elif fmt.startswith("reference16"):
        top.append('#include "mlp.hpp"')
        net, rows = backend["net"], []
        for i, L in enumerate(net["layers"]):
            o, n_in = L["w"].shape
            body += [_ints(f"critic_w{i}", "int16_t", L["w"]), _ints(f"critic_b{i}", "int32_t", L["b"])]
            rows.append(f"    {{{n_in}, {o}, critic_w{i}, critic_b{i}, {L['shift']}, {'true' if L['hidden'] else 'false'}}},")
            body.append(f'static_assert({max(o, n_in)} <= fusion::kMaxWidth, "critic layer {i} exceeds kMaxWidth");')
        body += ["inline constexpr fusion::Layer critic_layers[] = {", *rows, "};",
                 f"inline constexpr fusion::Net kCritic{{critic_layers, {len(rows)}, {net['out_exp']}}};", INPUT,
                 "inline double value(const float* f) {", f"    int32_t x[{net['in']}], acc[1];",
                 f"    for (int i = 0; i < {net['in']}; i++) x[i] = quantize(f[i]);",
                 "    fusion::forwardRaw(kCritic, x, acc);", "    return std::ldexp(double(acc[0]), -kCritic.outExp);", "}"]
    else:
        top.append('#include "mlp_wide.hpp"')
        net, rows = backend["net"], []
        for i, L in enumerate(net["layers"]):
            o, n_in = L["w"].shape
            body += [_ints(f"critic_w{i}", "int16_t", L["w"]), _ints(f"critic_b{i}", "int64_t", L["b"])]
            rows.append(f"    {{{n_in}, {o}, critic_w{i}, critic_b{i}, {L['shift']}, {'true' if L['hidden'] else 'false'}}},")
            body.append(f'static_assert({max(o, n_in)} <= fusion::wide::kMaxWidth, "critic layer {i} exceeds the wide buffers");')
        body += ["inline constexpr fusion::wide::Layer critic_layers[] = {", *rows, "};",
                 f"inline constexpr fusion::wide::Net kCritic{{critic_layers, {len(rows)}, {net['out_exp']}}};", INPUT,
                 "inline double value(const float* f) {", f"    int32_t x[{net['in']}];", "    int64_t acc[1];",
                 f"    for (int i = 0; i < {net['in']}; i++) x[i] = quantize(f[i]);",
                 "    fusion::wide::forwardRaw(kCritic, x, acc);", "    return fusion::wide::toReal(kCritic, acc[0]);", "}"]
    return "\n".join([*top, "namespace meow_rl_critic {", f'inline constexpr const char* kFormat = "{fmt}";',
                      f'inline constexpr const char* kDigest = "{backend["digest"]}";',
                      f'inline constexpr const char* kIdentity = "{I.digest(identity)}";', *body,
                      "}  // namespace meow_rl_critic", ""])
