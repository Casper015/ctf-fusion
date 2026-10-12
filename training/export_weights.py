"""Quantize one model of the matrix and emit standalone C++ headers.

    python -m training.export_weights --model n08 --policy runs/policy_n08 --value runs/value_n08 \
        --report reports/models/s/n08.json

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

Feature families (fusion/families.py): each run records the family it was
trained on; the export recomputes the family digest from the extractor and
refuses a mismatch. A p2/v2 model is exported exactly as before; a p3 or v3
model's header includes policy_p3.hpp / value_v3.hpp, uses their model
types and static_asserts its input widths, so new features cannot reach old
weights (or the reverse). One registry holds one policy and one value family.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from fusion import families, quant
from fusion.capacity import buffer_width
from fusion.mixture import ADAPTERS
from fusion.models import TARGET_COLS, PolicyNet, ValueNet
from fusion.paths import REPO, load_profile, served_counts, sha256
from fusion.schema import Layout
from fusion.schema import layout as feature_layout


def input_specs(layout: Layout, policy_family: str = "p2", value_family: str = "v2") -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(scale, lo, hi) of each net's input vector, in the order inference/policy.hpp (policy_p3.hpp) builds it."""
    def group(*names):
        parts = [(layout.scale(n), *layout.bounds(n)) for n in names]
        return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))

    combat = ["combat"] if policy_family == "p3" else []
    mover = group("global", "unit", "move", *combat)
    squad = group("global", "unit", "unit", "unit", *(["move"] * 15), *(combat * 15))
    cs, (clo, chi) = layout.scale("cand_f"), layout.bounds("cand_f")
    cols = [4 if c is None else c for c in TARGET_COLS]  # every hit bit shares hit_S's spec
    target = (cs[cols], clo[cols], chi[cols])
    lo, hi = layout.value_bounds(value_family)
    return {"mover": mover, "squad": squad, "act": mover, "target": target, "value": (layout.value_scale(value_family), lo, hi)}


# Run kinds that are never opponent P or board value V (fusion/decision_features.py: our own complete orders).
NOT_P_OR_V = ("own_action_scorer",)


def refuse_foreign_kind(cfg: dict, run) -> None:
    """An own-action scorer must never be exported or loaded as opponent P or legacy V."""
    if cfg.get("kind") in NOT_P_OR_V:
        raise SystemExit(f"{run}: kind {cfg['kind']!r} is our own action scorer, not opponent P or V; it cannot be "
                         "exported or loaded as either")


def run_config(run: Path, layout: Layout, kind: str = "policy") -> dict:
    """A run's config.json; refuses an own-action scorer and a run whose feature family the extractor does not reproduce."""
    cfg = json.loads((Path(run) / "config.json").read_text(encoding="utf-8"))
    refuse_foreign_kind(cfg, run)
    families.check(layout.schema, families.run_features(cfg, kind), str(run))
    return cfg


def load_policy(run: Path, layout: Layout) -> PolicyNet:
    cfg = run_config(run, layout, "policy")
    model = PolicyNet(layout, np.random.default_rng(0), cfg["hidden"], families.run_features(cfg, "policy")["family"])
    model.load(dict(np.load(run / "model.npz")))
    return model


def load_value(run: Path, layout: Layout) -> tuple[ValueNet, float]:
    cfg = run_config(run, layout, "value")
    model = ValueNet(layout, np.random.default_rng(0), cfg["hidden"], families.run_features(cfg, "value")["family"])
    st = dict(np.load(run / "model.npz"))
    model.load(st)
    return model, float(st.get("beta", 1.0))


def quantize_model(policy: PolicyNet, value: ValueNet, value_beta: float, layout: Layout,
                   max_width: int = quant.MAX_WIDTH) -> dict[str, dict]:
    """Integer nets; raises ValueError when a layer does not fit C++ buffers of `max_width`."""
    spec = input_specs(layout, getattr(policy, "family", "p2"), getattr(value, "family", "v2"))
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


def model_width(policy: PolicyNet, value: ValueNet) -> int:
    """Required input/output capacity without allocating quantized arrays."""
    return buffer_width(int(d) for net in (*policy.nets.values(), value.net) for w in net.W for d in w.shape)


def flat_arrays(nets: dict[str, dict]) -> dict[str, np.ndarray]:
    flat = {}
    for name, q in nets.items():
        flat |= quant.flatten(q, name)
    return flat


def schema_digest(layout: Layout) -> str:
    return hashlib.sha256(json.dumps(layout.schema, sort_keys=True).encode()).hexdigest()[:16]


# The full-schema digest every report and manifest recorded before feature families (spec v2, record layout 2).
# A report identity that carries it describes the families p2 and v2 (fusion/families.py LEGACY_DIGESTS).
LEGACY_SCHEMA_DIGEST = "d3813c50359fb9cf"


def model_features(policy: PolicyNet, value: ValueNet, layout: Layout) -> dict:
    return {"policy": families.record(layout.schema, policy.family), "value": families.record(layout.schema, value.family)}


def artifact_identity(model: str, policy_run, value_run, nets: dict, layout: Layout, adapter: str,
                      features: dict | None = None) -> dict:
    """What a gate report approves: this model id, these checkpoints, these integer weights
    (temperatures included), these feature families, this adapter.

    The legacy pair (p2, v2) keeps the identity of before families (spec_version and the full schema digest of
    layout 2), so reports made then still verify; any other pair is identified by its family digests."""
    out = {"model": model, "policy_sha256": sha256(Path(policy_run) / "model.npz"),
           "value_sha256": sha256(Path(value_run) / "model.npz"), "weights_digest": quant.digest(flat_arrays(nets))}
    names = {k: v["family"] for k, v in (features or {}).items()}
    if not features or names == families.LEGACY:
        return out | {"spec_version": layout.schema["spec_version"], "schema_digest": LEGACY_SCHEMA_DIGEST, "adapter": adapter}
    return out | {"features": {k: {"family": v["family"], "digest": v["digest"]} for k, v in features.items()}, "adapter": adapter}


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


# C++ names per family: (model type, extra header, input-width constant per net).
POLICY_CPP = {"p2": ("fusion::PolicyModel", None, {"mover": "fusion::kMoverIn", "squad": "fusion::kSquadIn", "act": "fusion::kMoverIn",
                                                   "target": "fusion::kTargetIn"}),
              "p3": ("fusion::PolicyModelP3", "policy_p3.hpp", {"mover": "fusion::kMoverInP3", "squad": "fusion::kSquadInP3",
                                                                "act": "fusion::kMoverInP3", "target": "fusion::kTargetIn"})}
VALUE_CPP = {"v2": ("fusion::ValueModel", None, {"value": "fusion::kV"}),
             "v3": ("fusion::ValueModelV3", "value_v3.hpp", {"value": "fusion::kV3"})}
FAMILY_VERSION = {"p3": "fusion::kPolicyFamilyP3 == 3", "v3": "fusion::kValueFamilyV3 == 3"}


def _emit_nets(out: list[str], nets: dict[str, dict], inputs: dict[str, str], check_inputs: bool, array_writer=_array) -> None:
    for name, q in nets.items():
        rows = []
        for i, layer in enumerate(q["layers"]):
            out.append(array_writer(f"{name}_w{i}", "int16_t", layer["w"]))
            out.append(array_writer(f"{name}_b{i}", "int32_t", layer["b"]))
            o, n_in = layer["w"].shape
            # The value accumulator updates one first-layer column per changed feature: a column-major
            # copy makes those columns contiguous for the SIMD kernel (same integers, same results).
            transposed = name == "value" and i == 0
            if transposed:
                out.append(array_writer(f"{name}_w{i}T", "int16_t", np.ascontiguousarray(layer["w"].T)))
            rows.append(f"    {{{n_in}, {o}, {name}_w{i}, {name}_b{i}, {layer['shift']}, {'true' if layer['hidden'] else 'false'}"
                        + (f", {name}_w{i}T" if transposed else "") + "},")
            out.append(f'static_assert({n_in} <= fusion::kMaxWidth && {o} <= fusion::kMaxWidth, "{name} layer {i} exceeds kMaxWidth");')
            if i == 0 and check_inputs:
                out.append(f'static_assert({n_in} == {inputs[name]}, "{name} input width differs from its feature family");')
        out += [f"inline constexpr fusion::Layer {name}_layers[] = {{", *rows, "};", ""]


def shared_value_header(name: str, value_net: dict, value_family: str, sources: dict,
                        generator: str = "training/export_mixed.py", array_writer=_array) -> str:
    """One value network shared by every model of a set (fusion::models::<name>::kValue), written once: a wide V is
    not copied into each model's header. Model headers alias it (header(shared_value=name))."""
    required = buffer_width([widest({"value": value_net})])
    out = [f"// Generated by {generator} on {time.strftime('%Y-%m-%d')}. Do not edit.",
           f"// Shared value network {name}. Sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()),
           f"// Feature family: value {value_family} (fusion/families.py).",
           f"// FUSION_MAX_WIDTH_REQUIRED={required}",
           f"#ifndef FUSION_MODEL_{name.upper()}_HPP", f"#define FUSION_MODEL_{name.upper()}_HPP",
           "#include <cstdint>", "", '#include "value.hpp"', *([f'#include "{VALUE_CPP[value_family][1]}"'] if VALUE_CPP[value_family][1] else []), "",
           *([f'static_assert({FAMILY_VERSION[value_family]}, "model trained on feature family {value_family}");'] if value_family in FAMILY_VERSION else []),
           "", f"namespace fusion::models::{name} {{", ""]
    _emit_nets(out, {"value": value_net}, VALUE_CPP[value_family][2], True, array_writer)
    out += [f"inline constexpr {VALUE_CPP[value_family][0]} kValue{{{{value_layers, {len(value_net['layers'])}, {value_net['out_exp']}}}}};",
            "", f"}}  // namespace fusion::models::{name}", f"#endif  // FUSION_MODEL_{name.upper()}_HPP", ""]
    return "\n".join(out)


def header(model: str, nets: dict[str, dict], alpha: dict[int, float], sources: dict, adapter: str = "joint-v2",
           spec_version: int | None = None, generator: str = "training/export_weights.py",
           policy_family: str = "p2", value_family: str = "v2", shared_value: tuple[str, dict] | None = None,
           array_writer=_array) -> str:
    """One model's header. With shared_value=(name, value net) the value arrays live in weights_<name>.hpp
    (shared_value_header) and this header aliases them; `nets` then holds the policy heads only."""
    spec_version = feature_layout().schema["spec_version"] if spec_version is None else spec_version
    legacy = (policy_family, value_family) == ("p2", "v2") and shared_value is None
    wide = widest(nets if shared_value is None else nets | {"value": shared_value[1]})
    required = buffer_width([wide])
    extra = [f'#include "{h}"' for h in (POLICY_CPP[policy_family][1], VALUE_CPP[value_family][1]) if h]
    if shared_value is not None:
        extra.append(f'#include "weights_{shared_value[0]}.hpp"')
    out = [f"// Generated by {generator} on {time.strftime('%Y-%m-%d')}. Do not edit.",
           f"// Model {model}. Sources: " + ", ".join(f"{k}={v}" for k, v in sources.items()),
           *([] if legacy else [f"// Feature families: policy {policy_family}, value {value_family} (fusion/families.py)."]),
           f"// FUSION_MAX_WIDTH_REQUIRED={required}",
           f"// Widest layer {wide}" + (f": build the bot with -DFUSION_MAX_WIDTH={wide} or more." if wide > quant.MAX_WIDTH else "."),
           f"#ifndef FUSION_MODEL_{model.upper()}_HPP", f"#define FUSION_MODEL_{model.upper()}_HPP",
           "#include <cstdint>", "", "// Needs inference/ on the include path (or the headers inlined before this one).",
           '#include "adapter.hpp"', '#include "policy.hpp"', '#include "value.hpp"', *extra, "",
           f"static_assert(fusion::kSpecVersion == {spec_version}, \"model trained on feature spec v{spec_version}\");",
           *[f'static_assert({FAMILY_VERSION[f]}, "model trained on feature family {f}");' for f in (policy_family, value_family)
             if f in FAMILY_VERSION], "",
           f"namespace fusion::models::{model} {{", "",
           f"inline constexpr int kRequiredMaxWidth = {required};", ""]
    if not legacy:
        out += [f"inline constexpr int kPolicyFamily = {policy_family[1:]};  // {policy_family}",
                f"inline constexpr int kValueFamily = {value_family[1:]};  // {value_family}", ""]
    inputs = POLICY_CPP[policy_family][2] | VALUE_CPP[value_family][2]
    _emit_nets(out, nets, inputs, not legacy, array_writer)
    net = lambda n: f"{{{n}_layers, {len(nets[n]['layers'])}, {nets[n]['out_exp']}}}"
    value_line = (f"inline constexpr {VALUE_CPP[value_family][0]} kValue{{{net('value')}}};" if shared_value is None else
                  f"inline constexpr const {VALUE_CPP[value_family][0]}& kValue = fusion::models::{shared_value[0]}::kValue;  // shared")
    out += [f"inline constexpr {POLICY_CPP[policy_family][0]} kPolicy{{{net('mover')}, {net('squad')}, {net('act')}, {net('target')}}};",
            value_line,
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


def manifest_families(entry: dict) -> tuple[str, str]:
    """(policy family, value family) of a manifest entry; entries from before families are (p2, v2)."""
    feats = entry.get("features") or {}
    return feats.get("policy", {}).get("family", "p2"), feats.get("value", {}).get("family", "v2")


def registry(out: Path, manifest: dict, generator: str = "training/export_weights.py") -> Path:
    """inference/models/registry.hpp: one dispatch table over every exported model (profiles decide).

    All models of one registry share one policy and one value family: ModelSet points at their types."""
    exported = sorted(manifest)
    fams = {manifest_families(manifest[m]) for m in exported}
    if len(fams) > 1:
        raise SystemExit(f"{out}: one registry holds one feature-family pair, found {sorted(fams)}; export them apart")
    pfam, vfam = fams.pop() if fams else ("p2", "v2")
    required = buffer_width([int(manifest[m].get("required_buffer_width", manifest[m].get("widest_layer", quant.MAX_WIDTH)))
                             for m in exported] or [quant.MAX_WIDTH])
    rows = []
    for t in range(16):
        model = None
        if 2 <= t <= 15:
            wanted = load_profile(t)["model"]
            model = wanted if wanted in manifest else ("general" if "general" in manifest else None)
        rows.append("    {nullptr, nullptr, nullptr, 0.0, 0}," if model is None else
                    f'    {{"{model}", &{model}::kPolicy, &{model}::kValue, {model}::kAlpha[{t}], {model}::kAdapter}},  // {t} teams')
    text = "\n".join([
        f"// Generated by {generator} from MANIFEST.json and profiles/. Do not edit.",
        "// Unified dispatch: a team count uses its dedicated model when exported, else the general model,",
        "// else none (nullptr: the bot keeps its legacy forecast and evaluation).",
        "#ifndef FUSION_MODELS_REGISTRY_HPP", "#define FUSION_MODELS_REGISTRY_HPP",
        f"// FUSION_MAX_WIDTH_REQUIRED={required}",
        "// This must precede the inference headers; an explicit smaller build fails their model assertions.",
        "#ifndef FUSION_MAX_WIDTH", f"#define FUSION_MAX_WIDTH {required}", "#endif",
        *[f'#include "{manifest[m]["header"]}"' for m in exported], "",
        "namespace fusion::models {", "", f"inline constexpr int kRegistryRequiredMaxWidth = {required};", "",
        f"struct ModelSet {{ const char* id; const {POLICY_CPP[pfam][0].removeprefix('fusion::')}* policy; "
        f"const {VALUE_CPP[vfam][0].removeprefix('fusion::')}* value; double alpha; int adapter; }};", "",
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
    ap.add_argument("--max-width", type=int, default=0,
                    help="capacity ceiling (0: infer from actual layer shapes); explicit smaller limits fail closed")
    ap.add_argument("--out", default=str(REPO / "inference" / "models"))
    a = ap.parse_args(argv)
    layout = feature_layout()
    policy = load_policy(Path(a.policy), layout)
    value, value_beta = load_value(Path(a.value), layout)
    if a.max_width < 0:
        raise SystemExit("--max-width must be positive, or 0 to infer capacity")
    a.max_width = a.max_width or model_width(policy, value)
    try:
        nets = quantize_model(policy, value, value_beta, layout, a.max_width)
    except ValueError as exc:
        raise SystemExit(f"cannot export: {exc}") from exc
    features = model_features(policy, value, layout)
    identity = artifact_identity(a.model, a.policy, a.value, nets, layout, a.adapter, features)
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
    hpp.write_text(header(a.model, nets, alpha, sources, a.adapter, layout.schema["spec_version"],
                          policy_family=policy.family, value_family=value.family), encoding="utf-8")
    np.savez(out / f"{stem}.npz", **flat_arrays(nets))
    manifest_path = out / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    clash = {m: manifest_families(e) for m, e in manifest.items() if m != a.model and manifest_families(e) != (policy.family, value.family)}
    if clash:
        raise SystemExit(f"{out} already holds models of other feature families {clash}: export into another --out")
    per_turn = 15 * macs(nets["mover"]) + macs(nets["squad"])
    manifest[a.model] = {
        "header": hpp.name, "header_sha256": sha256(hpp), "npz_sha256": sha256(out / f"{stem}.npz"),
        "identity": identity, "spec_version": layout.schema["spec_version"], "features": features, "adapter": a.adapter,
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
        "required_buffer_width": buffer_width([widest(nets)]),
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
