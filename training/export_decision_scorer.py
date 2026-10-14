"""Export an own-action scorer for the carrier hook (task T0044, docs/RL_PLAN.md sections 4 and 5).

    python -m training.export_decision_scorer --run runs/decision_mlp_s1 --selection reports/models/decision/SELECTION.json \\
        --model-name mlp --out inference/own_action_scorers/mlp_s1
    # a test export without a frozen selection (never deployable as selected)
    python -m training.export_decision_scorer --run <run> --lambda 0.5 --norm zscore --unselected --out build/scorer_test

Writes <out>/scorer_model.hpp (fusion::own_action::exported::kScorer, an inference/own_action_scorer.hpp ScorerModel)
and <out>/MANIFEST.json. This is the only export route of model kind own_action_scorer; training.export_weights and
training.export_mixed refuse it, and this command refuses every other kind, so a scorer can never be written over
or loaded as opponent P or legacy V. An output directory inside an opponent-P/V model directory (inference/models*)
is refused.

Bound into the header and the manifest, and checked here:
  * the run: kind own_action_scorer (config.json and model.npz), the feature identity the extractor provides now
    (fusion.decision_features.feature_identity: version, specs with their scales, block shapes), the dimensions, the
    sha256 of config.json and model.npz;
  * the rule: lambda and normalization from a frozen selection (training.evaluate_decisions select), whose own digest
    and whose recorded run hashes must match; or, with --unselected, values given here and marked "unselected";
  * the carrier: the validated carrier configuration of the run's training data (variant, status, opponent-P family
    and model directory, V on/off, the configuration's sha256). A run trained on data without one is refused unless
    --allow-unbound-carrier (marked "unbound"; the carrier hook refuses to build with it unless told to).
Weights are the trained float32 values, written exactly as hexadecimal float literals. No quantization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

from fusion import decision_features as F
from training import decision_dataset as D
from training.evaluate_decisions import NORMS, SELECTION_SCHEMA, canonical_sha
from training.train_decision_ranker import load_scorer

GENERATOR = "training/export_decision_scorer.py"
HEADER = "scorer_model.hpp"
MAX_WIDTH = 1024   # inference/own_action_scorer.hpp kScorerMaxWidth


def c_float(x: float) -> str:
    """A float32 value as an exact C++ literal."""
    v = float(np.float32(x))
    if not math.isfinite(v):
        raise SystemExit(f"non-finite weight {v}")
    return f"{v.hex()}f" if v != 0 else "0.0f"


def c_str(s: str) -> str:
    return json.dumps(str(s))


def refuse_model_dir(out: Path) -> None:
    parts = out.resolve().parts
    for i, part in enumerate(parts[:-1]):
        if part == "inference" and parts[i + 1].startswith("models"):
            raise SystemExit(f"{out}: inside an opponent-P/V model directory (inference/models*); a scorer has its own "
                             "directory, e.g. inference/own_action_scorers/<name>")


def rule_from_selection(path: Path, name: str, run: Path) -> dict:
    sel = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in sel.items() if k != "selection_sha256"}
    if sel.get("schema") != SELECTION_SCHEMA or canonical_sha(body) != sel.get("selection_sha256"):
        raise SystemExit(f"{path}: not a frozen selection, or changed after it was frozen")
    if name not in sel.get("models", {}):
        raise SystemExit(f"{path}: no model {name!r} (models: {sorted(sel.get('models', {}))})")
    m = sel["models"][name]
    if m["model_sha256"] != D.sha256_file(run / "model.npz") or m["config_sha256"] != D.sha256_file(run / "config.json"):
        raise SystemExit(f"{run} is not the run the selection froze as {name!r}")
    return {"lambda": float(m["lambda"]), "norm": m["norm"], "selection_sha256": sel["selection_sha256"],
            "selection": str(path), "model_name": name}


def carrier_binding(cfg: dict, allow_unbound: bool) -> dict:
    carrier = cfg.get("carrier") or {}
    conf = carrier.get("config")
    if not conf:
        if not allow_unbound:
            raise SystemExit("the run's training data has no validated carrier configuration (dataset built without "
                             "--carrier-config): pass --allow-unbound-carrier for a test export")
        return {"config_sha256": "unbound", "variant": "unbound", "status": "unbound", "p_family": "unbound",
                "p_models": "unbound", "value": -1, "binary_sha256": carrier.get("binary_sha256", [])}
    c = conf["config"]
    return {"config_sha256": conf["sha256"], "variant": c["variant"], "status": c["status"],
            "p_family": c["opponent_policy"]["family"], "p_models": c["opponent_policy"]["models"],
            "value": int(bool(c["value"]["enabled"])), "binary_sha256": carrier.get("binary_sha256", [])}


def layers_of(scorer) -> list[tuple[np.ndarray, np.ndarray]]:
    net = scorer.net
    return [(np.asarray(w, np.float32), np.asarray(b, np.float32)) for w, b in zip(net.W, net.b)]


def header(scorer, ident: dict, run_sha: dict, rule: dict, carrier: dict, cfg: dict) -> str:
    layers = layers_of(scorer)
    for w, _ in layers:
        if max(w.shape) > MAX_WIDTH:
            raise SystemExit(f"layer {w.shape} wider than {MAX_WIDTH} (inference/own_action_scorer.hpp kScorerMaxWidth)")
    lines = [f"// Generated by {GENERATOR} on {time.strftime('%Y-%m-%d')}. Do not edit.",
             f"// Own-action scorer ({scorer.model}{'' if not scorer.hidden else ' ' + ','.join(map(str, scorer.hidden))}), "
             f"kind {F.KIND}: never opponent P, never V.",
             "#ifndef FUSION_OWN_ACTION_SCORER_MODEL_HPP", "#define FUSION_OWN_ACTION_SCORER_MODEL_HPP",
             "// Needs inference/ on the include path (or the headers inlined before this one).",
             '#include "own_action_scorer.hpp"', "",
             f'static_assert(fusion::own_action::kRootDim == {cfg["root_dim"]} && fusion::own_action::kCandDim == {cfg["cand_dim"]}, '
             '"scorer dimensions");', "",
             "namespace fusion::own_action::exported {", ""]
    for i, (w, b) in enumerate(layers):
        lines.append(f"inline constexpr float W{i}[] = {{")
        flat = [c_float(x) for x in w.reshape(-1)]
        for j in range(0, len(flat), 8):
            lines.append("    " + ", ".join(flat[j:j + 8]) + ",")
        lines.append("};")
        lines.append(f"inline constexpr float B{i}[] = {{{', '.join(c_float(x) for x in b)}}};")
    lines.append("inline constexpr ScorerLayer kLayers[] = {")
    for i, (w, _) in enumerate(layers):
        lines.append(f"    {{{w.shape[1]}, {w.shape[0]}, W{i}, B{i}, {'true' if i + 1 < len(layers) else 'false'}}},")
    lines.append("};")
    mode = "ScorerInput::kCandidate" if scorer.model == "linear" else "ScorerInput::kRootAndCandidate"
    fields = [c_str(F.KIND), c_str(scorer.model), c_str(ident["version"]), c_str(ident["digest"]),
              str(cfg["root_dim"]), str(cfg["cand_dim"]), mode, "kLayers", str(len(layers)),
              c_str(run_sha["config_sha256"]), c_str(run_sha["model_sha256"]), c_str(carrier["config_sha256"]),
              c_str(carrier["variant"]), c_str(carrier["status"]), c_str(carrier["p_family"]), c_str(carrier["p_models"]),
              str(carrier["value"]), repr(float(rule["lambda"])), c_str(rule["norm"]), c_str(rule["selection_sha256"])]
    lines += ["inline constexpr ScorerModel kScorer{" + ", ".join(fields) + "};", "",
              "}  // namespace fusion::own_action::exported", "#endif  // FUSION_OWN_ACTION_SCORER_MODEL_HPP", ""]
    return "\n".join(lines)


def export(a) -> dict:
    run, out = Path(a.run), Path(a.out)
    refuse_model_dir(out)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if cfg.get("kind") != F.KIND:
        raise SystemExit(f"{run}: kind {cfg.get('kind')!r} is not {F.KIND}; opponent P and V use training.export_weights")
    scorer = load_scorer(run)          # refuses another kind and a feature identity the extractor lacks
    if cfg["root_dim"] != scorer.root_dim or cfg["cand_dim"] != scorer.cand_dim:
        raise SystemExit(f"{run}: dimensions in config.json differ from the model")
    run_sha = {"config_sha256": D.sha256_file(run / "config.json"), "model_sha256": D.sha256_file(run / "model.npz")}
    if a.selection:
        if a.unselected or a.lam is not None:
            raise SystemExit("--selection gives lambda and the normalization: drop --lambda/--unselected")
        rule = rule_from_selection(Path(a.selection), a.model_name, run)
    else:
        if not a.unselected or a.lam is None:
            raise SystemExit("without --selection, pass --lambda, --norm and --unselected (a test export)")
        if a.norm not in NORMS or not (a.lam >= 0 and math.isfinite(a.lam)):
            raise SystemExit(f"--norm must be one of {NORMS} and --lambda finite and non-negative")
        rule = {"lambda": float(a.lam), "norm": a.norm, "selection_sha256": "unselected", "selection": None, "model_name": None}
    carrier = carrier_binding(cfg, a.allow_unbound_carrier)
    text = header(scorer, cfg["features"], run_sha, rule, carrier, cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / HEADER).write_text(text, encoding="utf-8")
    manifest = {"schema": "ctf-fusion-own-action-scorer-export-1", "generator": GENERATOR, "kind": F.KIND,
                "model": scorer.model, "hidden": list(scorer.hidden), "features": cfg["features"],
                "root_dim": cfg["root_dim"], "cand_dim": cfg["cand_dim"],
                "run": {"path": str(run)} | run_sha, "rule": rule, "carrier": carrier,
                "dataset": cfg.get("dataset"), "quantization": "none (float32 weights, double accumulation)",
                "header": HEADER, "header_sha256": hashlib.sha256(text.encode()).hexdigest()}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[export_decision_scorer] {out / HEADER}: {scorer.model}, lambda {rule['lambda']} ({rule['norm']}, "
          f"{rule['selection_sha256'][:16]}), carrier {carrier['variant']} ({carrier['status']})")
    return manifest


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an own-action scorer run (training.train_decision_ranker)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--selection", help="frozen selection (training.evaluate_decisions select)")
    ap.add_argument("--model-name", default="mlp", help="the run's name in the selection")
    ap.add_argument("--lambda", dest="lam", type=float, help="test export only (with --unselected)")
    ap.add_argument("--norm", default="zscore")
    ap.add_argument("--unselected", action="store_true", help="a test export: lambda and norm not from a selection")
    ap.add_argument("--allow-unbound-carrier", action="store_true", help="a test export: no carrier configuration")
    return ap


def main(argv=None) -> dict:
    return export(parser().parse_args(argv))


if __name__ == "__main__":
    main()
