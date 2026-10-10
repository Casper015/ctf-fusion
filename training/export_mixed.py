"""Combine one tier's policy P with another tier's value V into a separate model set.

    python -m training.export_mixed --policy-tier xl --value-tier m     # -> inference/models_pxl_vm/
    python -m training.export_mixed --policy-tier xxl --value-tier m    # -> inference/models_pxxl_vm/
    python -m training.export_mixed --policy-tier xxl --value-dir inference/models_v3_4xl --out inference/models_pxxl_v3_4xl

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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-tier", choices=sorted(TIER_DIR))
    ap.add_argument("--value-tier", choices=sorted(TIER_DIR))
    ap.add_argument("--policy-dir", help="model-set directory (with MANIFEST.json) to take P from, instead of a tier")
    ap.add_argument("--value-dir", help="model-set directory (with MANIFEST.json) to take V from, instead of a tier")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    export(a.policy_tier, a.value_tier, Path(a.out) if a.out else None, a.policy_dir, a.value_dir)


if __name__ == "__main__":
    main()
