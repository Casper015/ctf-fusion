"""Quantize one model of the matrix and emit standalone C++ headers.

    python -m training.export_weights --model n08 --policy runs/policy_n08 --value runs/value_n08 \
        --report reports/n08.json

Writes, under inference/models/:
  weights_<model>.hpp  int16 weights as constexpr arrays (no runtime files), kAlpha per team count, kAdapter
  weights_<model>.npz  the same integers, for the Python emulation and the parity tests
  MANIFEST.json        per model: artifact identity, hashes, training sources, gate report, alpha, sizes
  registry.hpp         unified dispatch over every exported model: fusion::models::forTeams(teams)

Fitted temperatures are folded into the last layers, feature scales into the
first. kAlpha[t] is the held-out mixture weight for team count t when the
report marks it `use`, else 0: alpha 0 leaves the legacy forecast untouched,
so a model that does not beat the heuristic cannot change play.

The report must describe this exact artifact. evaluate_offline records an
identity (model id, checkpoint hashes, integer weight digest, feature schema,
adapter) and the exporter recomputes it; any difference stops the export, so
a retrained checkpoint cannot inherit an older positive report. Without
--report every alpha is 0.

Alpha gates the opponent policy only. No gate approves the value network yet
(kValueApproved = false): a bot that uses V needs its own switch and its own
evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fusion import quant
from fusion.mixture import ADAPTERS
from fusion.models import TARGET_COLS, PolicyNet, ValueNet
from fusion.paths import REPO, load_profile, served_counts, sha256
from fusion.schema import Layout
from fusion.schema import layout as feature_layout


def input_specs(layout: Layout) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(scale, lo, hi) of each net's input vector, in the order inference/policy.hpp builds it."""
    def group(*names):
        parts = [(layout.scale(n), *layout.bounds(n)) for n in names]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    mover = group("global", "unit", "move")
    squad = group("global", "unit", "unit", "unit", *(["move"] * 15))
    cs, (clo, chi) = layout.scale("cand_f"), layout.bounds("cand_f")
    cols = [4 if c is None else c for c in TARGET_COLS]  # every hit bit shares hit_S's spec
    target = (cs[cols], clo[cols], chi[cols])
    lo, hi = layout.value_bounds()
    return {"mover": mover, "squad": squad, "act": mover, "target": target, "value": (layout.value_scale(), lo, hi)}


def run_config(run: Path, layout: Layout) -> dict:
    """A run's config.json; refuses a run trained on another feature spec."""
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    if cfg.get("spec_version") != layout.schema["spec_version"]:
        raise SystemExit(f"{run} was trained on feature spec v{cfg.get('spec_version')}, "
                         f"the extractor is v{layout.schema['spec_version']}: retrain it")
    return cfg


def load_policy(run: Path, layout: Layout) -> PolicyNet:
    cfg = run_config(run, layout)
    model = PolicyNet(layout, np.random.default_rng(0), cfg["hidden"])
    model.load(dict(np.load(run / "model.npz")))
    return model


def load_value(run: Path, layout: Layout) -> tuple[ValueNet, float]:
    cfg = run_config(run, layout)
    model = ValueNet(layout, np.random.default_rng(0), cfg["hidden"])
    st = dict(np.load(run / "model.npz"))
    model.load(st)
    return model, float(st.get("beta", 1.0))


def quantize_model(policy: PolicyNet, value: ValueNet, value_beta: float, layout: Layout,
                   max_width: int = quant.MAX_WIDTH) -> dict[str, dict]:
    """Integer nets; raises ValueError when a layer does not fit C++ buffers of `max_width`."""
    spec = input_specs(layout)
    beta = policy.beta
    nets = {
        "mover": quant.quantize(policy.mover, *spec["mover"], out_mult=beta["joint"]),
        "squad": quant.quantize(policy.squad, *spec["squad"], out_mult=beta["joint"]),
        "act": quant.quantize(policy.act, *spec["act"], out_mult=beta["act"]),
        "target": quant.quantize(policy.target, *spec["target"], out_mult=beta["target"]),
        "value": quant.quantize(value.net, *spec["value"], out_rows=[0], out_mult=value_beta),
    }
    for name, q in nets.items():
        quant.check_widths(q, name, max_width)
    return nets


def flat_arrays(nets: dict[str, dict]) -> dict[str, np.ndarray]:
    flat = {}
    for name, q in nets.items():
        flat |= quant.flatten(q, name)
    return flat


def schema_digest(layout: Layout) -> str:
    return hashlib.sha256(json.dumps(layout.schema, sort_keys=True).encode()).hexdigest()[:16]


def artifact_identity(model: str, policy_run, value_run, nets: dict, layout: Layout, adapter: str) -> dict:
    """What a gate report approves: this model id, these checkpoints, these integer weights
    (temperatures included), this feature schema, this adapter."""
    return {"model": model, "policy_sha256": sha256(Path(policy_run) / "model.npz"),
            "value_sha256": sha256(Path(value_run) / "model.npz"), "weights_digest": quant.digest(flat_arrays(nets)),
            "spec_version": layout.schema["spec_version"], "schema_digest": schema_digest(layout), "adapter": adapter}


def verified_gates(report: dict, identity: dict) -> dict[int, dict]:
    """The report's per-count gates, if it was produced for exactly this artifact; else stop (fail closed)."""
    got = report.get("identity")
    if not got:
        raise SystemExit("the report carries no artifact identity (made before identities existed): rerun evaluate_offline")
    diff = [k for k in identity if got.get(k) != identity[k]]
    if diff:
        raise SystemExit("the report was produced for a different artifact (" + ", ".join(
            f"{k}: report {got.get(k)} vs export {identity[k]}" for k in diff) + "); rerun evaluate_offline on these runs")
    return {int(t): v for t, v in report.get("alpha", {}).items()}


def _array(name: str, ctype: str, values: np.ndarray) -> str:
    flat = [str(int(v)) for v in np.asarray(values).reshape(-1)]
    lines = [", ".join(flat[i:i + 24]) for i in range(0, len(flat), 24)]
    return f"inline constexpr {ctype} {name}[] = {{\n    " + ",\n    ".join(lines) + "\n};\n"


def widest(nets: dict[str, dict]) -> int:
    return max(max(layer["w"].shape) for q in nets.values() for layer in q["layers"])


def header(model: str, nets: dict[str, dict], alpha: dict[int, float], sources: dict, adapter: str = "joint-v2",
           spec_version: int | None = None) -> str:
    spec_version = feature_layout().schema["spec_version"] if spec_version is None else spec_version
    wide = widest(nets)
    out = [f"// Generated by training/export_weights.py on {time.strftime('%Y-%m-%d')}. Do not edit.",
           f"// Model {model}. Sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()),
           f"// Widest layer {wide}" + (f": build the bot with -DFUSION_MAX_WIDTH={wide} or more." if wide > quant.MAX_WIDTH else "."),
           f"#ifndef FUSION_MODEL_{model.upper()}_HPP", f"#define FUSION_MODEL_{model.upper()}_HPP",
           "#include <cstdint>", "", "// Needs inference/ on the include path (or the headers inlined before this one).",
           '#include "adapter.hpp"', '#include "policy.hpp"', '#include "value.hpp"', "",
           f"static_assert(fusion::kSpecVersion == {spec_version}, \"model trained on feature spec v{spec_version}\");", "",
           f"namespace fusion::models::{model} {{", ""]
    for name, q in nets.items():
        rows = []
        for i, layer in enumerate(q["layers"]):
            out.append(_array(f"{name}_w{i}", "int16_t", layer["w"]))
            out.append(_array(f"{name}_b{i}", "int32_t", layer["b"]))
            o, n_in = layer["w"].shape
            # The value accumulator updates one first-layer column per changed feature: a column-major
            # copy makes those columns contiguous for the SIMD kernel (same integers, same results).
            transposed = name == "value" and i == 0
            if transposed:
                out.append(_array(f"{name}_w{i}T", "int16_t", np.ascontiguousarray(layer["w"].T)))
            rows.append(f"    {{{n_in}, {o}, {name}_w{i}, {name}_b{i}, {layer['shift']}, {'true' if layer['hidden'] else 'false'}"
                        + (f", {name}_w{i}T" if transposed else "") + "},")
            out.append(f'static_assert({n_in} <= fusion::kMaxWidth && {o} <= fusion::kMaxWidth, "{name} layer {i} exceeds kMaxWidth");')
        out += [f"inline constexpr fusion::Layer {name}_layers[] = {{", *rows, "};", ""]
    net = lambda n: f"{{{n}_layers, {len(nets[n]['layers'])}, {nets[n]['out_exp']}}}"
    out += [f"inline constexpr fusion::PolicyModel kPolicy{{{net('mover')}, {net('squad')}, {net('act')}, {net('target')}}};",
            f"inline constexpr fusion::ValueModel kValue{{{net('value')}}};",
            "// Mixture weight of q_theta against the legacy forecast by team count; 0 = legacy behavior.",
            "inline constexpr double kAlpha[16] = {" + ", ".join(repr(float(alpha.get(t, 0.0))) for t in range(16)) + "};",
            f"// The adapter kAlpha was fitted for (inference/adapter.hpp); sample with exactly this one.",
            f"inline constexpr int kAdapter = {ADAPTERS[adapter]};  // {adapter}",
            "// No gate approves the value network yet: a bot must not let V change play on this model's word.",
            "inline constexpr bool kValueApproved = false;",
            "", f"}}  // namespace fusion::models::{model}", f"#endif  // FUSION_MODEL_{model.upper()}_HPP", ""]
    return "\n".join(out)


def macs(q: dict) -> int:
    return int(sum(layer["w"].size for layer in q["layers"]))


def registry(out: Path, manifest: dict) -> Path:
    """inference/models/registry.hpp: one dispatch table over every exported model (profiles decide)."""
    exported = sorted(manifest)
    rows = []
    for t in range(16):
        model = None
        if 2 <= t <= 15:
            wanted = load_profile(t)["model"]
            model = wanted if wanted in manifest else ("general" if "general" in manifest else None)
        rows.append("    {nullptr, nullptr, nullptr, 0.0, 0}," if model is None else
                    f'    {{"{model}", &{model}::kPolicy, &{model}::kValue, {model}::kAlpha[{t}], {model}::kAdapter}},  // {t} teams')
    text = "\n".join([
        "// Generated by training/export_weights.py from MANIFEST.json and profiles/. Do not edit.",
        "// Unified dispatch: a team count uses its dedicated model when exported, else the general model,",
        "// else none (nullptr: the bot keeps its legacy forecast and evaluation).",
        "#ifndef FUSION_MODELS_REGISTRY_HPP", "#define FUSION_MODELS_REGISTRY_HPP",
        *[f'#include "{manifest[m]["header"]}"' for m in exported], "",
        "namespace fusion::models {", "",
        "struct ModelSet { const char* id; const PolicyModel* policy; const ValueModel* value; double alpha; int adapter; };", "",
        "inline const ModelSet* forTeams(int teams) {",
        "    static const ModelSet table[16] = {", *rows, "    };",
        "    return teams >= 2 && teams <= 15 && table[teams].policy ? &table[teams] : nullptr;", "}", "",
        "}  // namespace fusion::models", "#endif  // FUSION_MODELS_REGISTRY_HPP", ""])
    path = out / "registry.hpp"
    path.write_text(text, encoding="utf-8")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="model id from profiles/: n02 ... n15 or general")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--value", required=True)
    ap.add_argument("--report", help="JSON report from training.evaluate_offline for these runs: alpha and the use gate per team count")
    ap.add_argument("--adapter", choices=sorted(ADAPTERS), default="joint-v2",
                    help="deployment adapter (inference/adapter.hpp); the report must have been fitted for the same one")
    ap.add_argument("--max-width", type=int, default=quant.MAX_WIDTH,
                    help="widest layer allowed; above 256 the bot must be built with -DFUSION_MAX_WIDTH (tools/capacity_bench.py)")
    ap.add_argument("--out", default=str(REPO / "inference" / "models"))
    a = ap.parse_args(argv)
    layout = feature_layout()
    policy = load_policy(Path(a.policy), layout)
    value, value_beta = load_value(Path(a.value), layout)
    try:
        nets = quantize_model(policy, value, value_beta, layout, a.max_width)
    except ValueError as exc:
        raise SystemExit(f"cannot export: {exc}") from exc
    identity = artifact_identity(a.model, a.policy, a.value, nets, layout, a.adapter)
    alpha, gates, report = {}, {}, None
    if a.report:
        report = json.loads(Path(a.report).read_text(encoding="utf-8"))
        gates = verified_gates(report, identity)
        alpha = {t: float(v["alpha"]) for t, v in gates.items() if v.get("use") and v.get("alpha") is not None}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sources = {"policy": Path(a.policy).name, "value": Path(a.value).name}
    stem = f"weights_{a.model}"
    hpp = out / f"{stem}.hpp"
    hpp.write_text(header(a.model, nets, alpha, sources, a.adapter, layout.schema["spec_version"]), encoding="utf-8")
    np.savez(out / f"{stem}.npz", **flat_arrays(nets))
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    per_turn = 15 * macs(nets["mover"]) + macs(nets["squad"])
    manifest[a.model] = {
        "header": hpp.name, "header_sha256": sha256(hpp), "npz_sha256": sha256(out / f"{stem}.npz"),
        "identity": identity, "spec_version": layout.schema["spec_version"], "adapter": a.adapter,
        "alpha": alpha, "gates": gates, "value_approved": False,
        "gate_report": None if report is None else {
            "path": str(a.report), "split": report.get("split"), "fit_split": report.get("fit_split"),
            "dataset": report.get("identity", {}).get("dataset"), "baseline": report.get("identity", {}).get("baseline")},
        "served_team_counts": served_counts(a.model), "beta": policy.beta | {"value": value_beta},
        "sources": {k: {"run": str(Path(v)), "model_sha256": sha256(Path(v) / "model.npz"),
                        "config": json.loads((Path(v) / "config.json").read_text(encoding="utf-8"))} for k, v in
                    (("policy", a.policy), ("value", a.value))},
        "params": {k: int(sum(l["w"].size + l["b"].size for l in q["layers"])) for k, q in nets.items()},
        "widest_layer": widest(nets),
        "macs": {k: macs(q) for k, q in nets.items()} | {"joint_policy_per_team": per_turn},
        "accumulator_bound": {k: max(l["bound"] for l in q["layers"]) for k, q in nets.items()},
        "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    registry(out, manifest)
    print(f"[export] {hpp} ({hpp.stat().st_size // 1024} KiB), joint policy {per_turn} MACs per team, "
          f"value {macs(nets['value'])} MACs per state; alpha used for team counts {sorted(alpha) or 'none'}")


if __name__ == "__main__":
    main()
