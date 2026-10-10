"""Combine one tier's policy P with another tier's value V into a separate model set.

    python -m training.export_mixed --policy-tier xl --value-tier m     # -> inference/models_pxl_vm/
    python -m training.export_mixed --policy-tier xxl --value-tier m    # -> inference/models_pxxl_vm/
    python -m training.export_mixed --policy-tier xxl --value-dir inference/models_v3_4xl --out inference/models_pxxl_v3_4xl
    python -m training.export_mixed --policy-tier xxl --value-run runs/v_pilot_D_s1 --out inference/models_pxxl_vpD
    python -m training.export_mixed --policy-run runs/p_feat_p3_xxl_s1 --value-tier m --out inference/models_pf_p3xxl_vm

Controlled V ablations hold V fixed while P changes size. The exported tiers
cannot be combined in one binary: every tier uses the same include guards and
namespaces (fusion::models::general, ...). This generator reads the committed
integer arrays (weights_<model>.npz) of both tiers, takes the four policy heads
(mover, squad, act, target) from the policy tier and the value net from the
value tier, checks that each array is byte-identical to its source, and writes
a complete model set with the exporter's own header and registry code: one
registry per binary, no clash. Nothing is retrained or requantized; the policy
alpha comes from the policy tier's gate and V stays unapproved.

MANIFEST.json records, per model, both sources (tier, header and npz SHA-256,
weights digest), the digests of the copied policy and value arrays, and the
buffer width; tests/test_mixed_export.py re-checks it.

Feature families (fusion/families.py): each side keeps the family it was
trained on (sources from before families are p2 / v2), and both must still be
reproduced by the extractor (same digest), so a frozen P can be paired with a
V of a newer family without retraining P. The header and registry use the
pair's types and width checks; legacy pairs are written exactly as before.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fusion import families, quant
from fusion.capacity import buffer_width
from fusion.paths import REPO, served_counts, sha256
from training.export_weights import header, macs, manifest_families, registry, widest

POLICY_HEADS = ("mover", "squad", "act", "target")
TIER_DIR = {"s": "models", "m": "models_m", "l": "models_l", "xl": "models_xl", "xxl": "models_xxl"}
GENERATOR = "training/export_mixed.py"


def head_arrays(npz: dict, head: str) -> dict:
    return {k: npz[k] for k in npz if k.split(".", 1)[0] == head}


def heads_digest(arrays: dict) -> str:
    return quant.digest(arrays)


def mixed_dir(policy_tier: str, value_tier: str) -> Path:
    return REPO / "inference" / f"models_p{policy_tier}_v{value_tier}"


def source_dir(tier: str | None, path: str | Path | None) -> Path:
    if (tier is None) == (path is None):
        raise SystemExit("name each source by a tier or by a model-set directory, not both")
    return REPO / "inference" / TIER_DIR[tier] if tier else Path(path)


def side_features(entry: dict, side: str) -> dict:
    """The `features` of one side of a source manifest entry (sources from before families: the legacy family)."""
    feats = (entry.get("features") or {}).get(side)
    if feats:
        return feats
    name = families.LEGACY[side]
    return {"family": name, "digest": families.LEGACY_DIGESTS[name], "layout_version": 2}


def export(policy_tier: str | None, value_tier: str | None, out: Path | None = None,
           policy_dir: str | Path | None = None, value_dir: str | Path | None = None) -> Path:
    from fusion.native import schema  # noqa: PLC0415
    src_p, src_v = source_dir(policy_tier, policy_dir), source_dir(value_tier, value_dir)
    name_p = TIER_DIR[policy_tier] if policy_tier else src_p.name
    name_v = TIER_DIR[value_tier] if value_tier else src_v.name
    man_p = json.loads((src_p / "MANIFEST.json").read_text(encoding="utf-8"))
    man_v = json.loads((src_v / "MANIFEST.json").read_text(encoding="utf-8"))
    if out is None:
        if not (policy_tier and value_tier):
            raise SystemExit("--out is required when a source is a directory")
        out = mixed_dir(policy_tier, value_tier)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for model in sorted(man_p):
        if model not in man_v:
            raise SystemExit(f"model {model} is in tier {policy_tier} but not in tier {value_tier}")
        zp = dict(np.load(src_p / f"weights_{model}.npz"))
        zv = dict(np.load(src_v / f"weights_{model}.npz"))
        pol = {k: v for h in POLICY_HEADS for k, v in head_arrays(zp, h).items()}
        val = head_arrays(zv, "value")
        flat = pol | val
        nets = {h: quant.unflatten(flat, h) for h in (*POLICY_HEADS, "value")}
        for k, v in flat.items():                      # the copy is exact, array by array
            ref = zp[k] if k in pol else zv[k]
            if v.dtype != ref.dtype or v.shape != ref.shape or not np.array_equal(v, ref):
                raise SystemExit(f"{model}: array {k} differs from its source")
        alpha = {int(t): float(a) for t, a in man_p[model].get("alpha", {}).items()}
        adapter = man_p[model].get("adapter", "joint-v2")
        feats = {"policy": side_features(man_p[model], "policy"), "value": side_features(man_v[model], "value")}
        for side, src in (("policy", src_p), ("value", src_v)):
            families.check(schema(), feats[side], f"{src.name}/{model} ({side})")
        pfam, vfam = feats["policy"]["family"], feats["value"]["family"]
        sources = {"policy": f"{name_p}/{model}", "value": f"{name_v}/{model}"}
        stem = f"weights_{model}"
        hpp = out / f"{stem}.hpp"
        hpp.write_text(header(model, nets, alpha, sources, adapter, man_p[model].get("spec_version"), GENERATOR,
                              policy_family=pfam, value_family=vfam), encoding="utf-8")
        np.savez(out / f"{stem}.npz", **flat)
        manifest[model] = {
            "header": hpp.name, "header_sha256": sha256(hpp), "npz_sha256": sha256(out / f"{stem}.npz"),
            "generator": GENERATOR, "spec_version": man_p[model].get("spec_version"), "adapter": adapter,
            **({} if (pfam, vfam) == ("p2", "v2") else {"features": feats}),
            "alpha": alpha, "value_approved": False, "served_team_counts": served_counts(model),
            "sources": {
                "policy": {"tier": name_p, "header_sha256": man_p[model]["header_sha256"],
                           "npz_sha256": man_p[model]["npz_sha256"], "weights_digest": man_p[model]["identity"]["weights_digest"],
                           "heads": list(POLICY_HEADS)},
                "value": {"tier": name_v, "header_sha256": man_v[model]["header_sha256"],
                          "npz_sha256": man_v[model]["npz_sha256"], "weights_digest": man_v[model]["identity"]["weights_digest"],
                          "heads": ["value"]}},
            "policy_digest": heads_digest(pol), "value_digest": heads_digest(val),
            "params": {k: int(sum(l["w"].size + l["b"].size for l in q["layers"])) for k, q in nets.items()},
            "widest_layer": widest(nets), "required_buffer_width": buffer_width([widest(nets)]),
            "macs": {k: macs(q) for k, q in nets.items()} | {"joint_policy_per_team": 15 * macs(nets["mover"]) + macs(nets["squad"])},
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    registry(out, manifest, GENERATOR)
    shown = out.relative_to(REPO) if out.resolve().is_relative_to(REPO) else out
    print(f"[export_mixed] {shown}: {len(manifest)} models, policy from {name_p}, "
          f"value from {name_v}, buffer width {max(m['required_buffer_width'] for m in manifest.values())}, "
          f"families {sorted({manifest_families(m) for m in manifest.values()})}")
    return out


def quantize_run_policy(run: Path, layout, max_width: int) -> tuple[dict, dict]:
    """The four policy heads of a training run, quantized as training.export_weights does (temperatures folded)."""
    from training.export_weights import input_specs, load_policy  # noqa: PLC0415
    policy = load_policy(Path(run), layout)
    spec = input_specs(layout, policy.family)
    beta = policy.beta
    mult = {"mover": beta["joint"], "squad": beta["joint"], "act": beta["act"], "target": beta["target"]}
    nets = {h: quant.quantize(getattr(policy, h), *spec[h], out_mult=mult[h]) for h in POLICY_HEADS}
    for h, q in nets.items():
        quant.check_widths(q, h, max_width)
    return nets, families.record(layout.schema, policy.family)


def quantize_run_value(run: Path, layout, max_width: int) -> tuple[dict, dict]:
    from training.export_weights import input_specs, load_value  # noqa: PLC0415
    value, beta = load_value(Path(run), layout)
    q = quant.quantize(value.net, *input_specs(layout, "p2", value.family)["value"], out_rows=[0], out_mult=beta)
    quant.check_widths(q, "value", max_width)
    return q, families.record(layout.schema, value.family)


def export_runs(out: Path, policy_tier: str | None = None, policy_dir=None, policy_run=None, model_id: str = "general",
                value_tier: str | None = None, value_dir=None, value_run=None, shared_name: str = "shared_value",
                max_width: int = 8192) -> Path:
    """A model set from trained runs and/or committed sets: a frozen P (tier, set or run) with a V (tier, set or run).

    A value run is quantized once and written once (weights_<shared_name>.hpp); every model's header aliases it,
    so a wide V is not copied per team count. A policy run gives one model (model_id; the registry serves every
    team count from `general`). Alpha is empty: the P-only carriers do not use it, and no gate report exists for
    a pairing. V stays unapproved."""
    from fusion.schema import layout as feature_layout  # noqa: PLC0415
    from training.export_weights import shared_value_header  # noqa: PLC0415
    L = feature_layout()
    out.mkdir(parents=True, exist_ok=True)
    policies = {}
    if policy_run:
        nets, feats = quantize_run_policy(Path(policy_run), L, max_width)
        flat = {k: v for h in POLICY_HEADS for k, v in quant.flatten(nets[h], h).items()}
        policies[model_id] = {"nets": nets, "flat": flat, "features": feats, "alpha": {}, "adapter": "joint-v2",
                              "source": {"run": str(policy_run), "model_sha256": sha256(Path(policy_run) / "model.npz"),
                                         "weights_digest": quant.digest(flat), "heads": list(POLICY_HEADS)}}
    else:
        src = source_dir(policy_tier, policy_dir)
        man = json.loads((src / "MANIFEST.json").read_text(encoding="utf-8"))
        for model in sorted(man):
            z = dict(np.load(src / f"weights_{model}.npz"))
            flat = {k: v for h in POLICY_HEADS for k, v in head_arrays(z, h).items()}
            feats = side_features(man[model], "policy")
            families.check(L.schema, feats, f"{src.name}/{model} (policy)")
            policies[model] = {"nets": {h: quant.unflatten(flat, h) for h in POLICY_HEADS}, "flat": flat, "features": feats,
                               "alpha": {int(t): float(a) for t, a in man[model].get("alpha", {}).items()},
                               "adapter": man[model].get("adapter", "joint-v2"),
                               "source": {"tier": src.name, "header_sha256": man[model]["header_sha256"], "npz_sha256": man[model]["npz_sha256"],
                                          "weights_digest": man[model]["identity"]["weights_digest"], "heads": list(POLICY_HEADS)}}
    shared, value_src, vfeats = None, None, None
    if value_run:
        q, vfeats = quantize_run_value(Path(value_run), L, max_width)
        vflat = quant.flatten(q, "value")
        value_src = {"run": str(value_run), "model_sha256": sha256(Path(value_run) / "model.npz"), "weights_digest": quant.digest(vflat),
                     "heads": ["value"], "shared": f"weights_{shared_name}.hpp"}
        shared = (shared_name, q)
        (out / f"weights_{shared_name}.hpp").write_text(shared_value_header(shared_name, q, vfeats["family"], {"value": Path(value_run).name},
                                                                            GENERATOR), encoding="utf-8")
        np.savez(out / f"weights_{shared_name}.npz", **vflat)
    else:
        vsrc = source_dir(value_tier, value_dir)
        man_v = json.loads((vsrc / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest = {}
    for model, p in policies.items():
        if shared is None:
            if model not in man_v:
                raise SystemExit(f"model {model} has no value network in {vsrc}")
            zv = dict(np.load(vsrc / f"weights_{model}.npz"))
            vflat_m = head_arrays(zv, "value")
            feats_v = side_features(man_v[model], "value")
            families.check(L.schema, feats_v, f"{vsrc.name}/{model} (value)")
            nets = p["nets"] | {"value": quant.unflatten(vflat_m, "value")}
            vsource = {"tier": vsrc.name, "header_sha256": man_v[model]["header_sha256"], "npz_sha256": man_v[model]["npz_sha256"],
                       "weights_digest": man_v[model]["identity"]["weights_digest"], "heads": ["value"]}
            flat = p["flat"] | vflat_m
        else:
            nets, feats_v, vsource, flat = p["nets"], vfeats, value_src, p["flat"]
        pfam, vfam = p["features"]["family"], feats_v["family"]
        all_nets = nets if shared is None else nets | {"value": shared[1]}
        stem = f"weights_{model}"
        hpp = out / f"{stem}.hpp"
        hpp.write_text(header(model, nets, p["alpha"], {"policy": p["source"].get("tier") or Path(p["source"]["run"]).name,
                                                        "value": vsource.get("tier") or Path(vsource["run"]).name},
                              p["adapter"], L.schema["spec_version"], GENERATOR, policy_family=pfam, value_family=vfam,
                              shared_value=shared), encoding="utf-8")
        np.savez(out / f"{stem}.npz", **flat)
        manifest[model] = {
            "header": hpp.name, "header_sha256": sha256(hpp), "npz_sha256": sha256(out / f"{stem}.npz"), "generator": GENERATOR,
            "spec_version": L.schema["spec_version"], "adapter": p["adapter"], "features": {"policy": p["features"], "value": feats_v},
            "alpha": p["alpha"], "value_approved": False, "served_team_counts": served_counts(model),
            "sources": {"policy": p["source"], "value": vsource},
            **({"shared_value": {"header": f"weights_{shared[0]}.hpp", "sha256": sha256(out / f"weights_{shared[0]}.hpp")}} if shared else {}),
            "params": {k: int(sum(l["w"].size + l["b"].size for l in q["layers"])) for k, q in all_nets.items()},
            "widest_layer": widest(all_nets), "required_buffer_width": buffer_width([widest(all_nets)]),
            "macs": {k: macs(q) for k, q in all_nets.items()} | {"joint_policy_per_team": 15 * macs(all_nets["mover"]) + macs(all_nets["squad"])},
            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    registry(out, manifest, GENERATOR)
    print(f"[export_mixed] {out}: {len(manifest)} models, families {sorted({manifest_families(m) for m in manifest.values()})}, "
          f"buffer width {max(m['required_buffer_width'] for m in manifest.values())}" + (f", shared V {shared[0]}" if shared else ""))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-tier", choices=sorted(TIER_DIR))
    ap.add_argument("--value-tier", choices=sorted(TIER_DIR))
    ap.add_argument("--policy-dir", help="model-set directory (with MANIFEST.json) to take P from, instead of a tier")
    ap.add_argument("--value-dir", help="model-set directory (with MANIFEST.json) to take V from, instead of a tier")
    ap.add_argument("--policy-run", help="a policy training run (one model, --model-id) instead of a tier or set")
    ap.add_argument("--model-id", default="general", help="model id of --policy-run (general serves every team count)")
    ap.add_argument("--value-run", help="a value training run, quantized once and shared by every model of the set")
    ap.add_argument("--shared-name", default="shared_value")
    ap.add_argument("--max-width", type=int, default=8192, help="buffer ceiling for quantizing runs")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    if a.policy_run or a.value_run:
        if not a.out:
            raise SystemExit("--out is required when a source is a training run")
        if a.policy_run and (a.policy_tier or a.policy_dir) or a.value_run and (a.value_tier or a.value_dir):
            raise SystemExit("name each side once: a tier, a set directory or a run")
        export_runs(Path(a.out), a.policy_tier, a.policy_dir, a.policy_run, a.model_id, a.value_tier, a.value_dir, a.value_run,
                    a.shared_name, a.max_width)
        return
    export(a.policy_tier, a.value_tier, Path(a.out) if a.out else None, a.policy_dir, a.value_dir)


if __name__ == "__main__":
    main()
