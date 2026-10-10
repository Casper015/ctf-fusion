"""Combine one tier's policy P with another tier's value V into a separate model set.

    python -m training.export_mixed --policy-tier xl --value-tier m     # -> inference/models_pxl_vm/
    python -m training.export_mixed --policy-tier xxl --value-tier m    # -> inference/models_pxxl_vm/

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
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fusion import quant
from fusion.capacity import buffer_width
from fusion.paths import REPO, served_counts, sha256
from training.export_weights import header, macs, registry, widest

POLICY_HEADS = ("mover", "squad", "act", "target")
TIER_DIR = {"s": "models", "m": "models_m", "l": "models_l", "xl": "models_xl", "xxl": "models_xxl"}
GENERATOR = "training/export_mixed.py"


def head_arrays(npz: dict, head: str) -> dict:
    return {k: npz[k] for k in npz if k.split(".", 1)[0] == head}


def heads_digest(arrays: dict) -> str:
    return quant.digest(arrays)


def mixed_dir(policy_tier: str, value_tier: str) -> Path:
    return REPO / "inference" / f"models_p{policy_tier}_v{value_tier}"


def export(policy_tier: str, value_tier: str, out: Path | None = None) -> Path:
    src_p, src_v = REPO / "inference" / TIER_DIR[policy_tier], REPO / "inference" / TIER_DIR[value_tier]
    man_p = json.loads((src_p / "MANIFEST.json").read_text(encoding="utf-8"))
    man_v = json.loads((src_v / "MANIFEST.json").read_text(encoding="utf-8"))
    out = out or mixed_dir(policy_tier, value_tier)
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
        sources = {"policy": f"{TIER_DIR[policy_tier]}/{model}", "value": f"{TIER_DIR[value_tier]}/{model}"}
        stem = f"weights_{model}"
        hpp = out / f"{stem}.hpp"
        hpp.write_text(header(model, nets, alpha, sources, adapter, man_p[model].get("spec_version"), GENERATOR), encoding="utf-8")
        np.savez(out / f"{stem}.npz", **flat)
        manifest[model] = {
            "header": hpp.name, "header_sha256": sha256(hpp), "npz_sha256": sha256(out / f"{stem}.npz"),
            "generator": GENERATOR, "spec_version": man_p[model].get("spec_version"), "adapter": adapter,
            "alpha": alpha, "value_approved": False, "served_team_counts": served_counts(model),
            "sources": {
                "policy": {"tier": TIER_DIR[policy_tier], "header_sha256": man_p[model]["header_sha256"],
                           "npz_sha256": man_p[model]["npz_sha256"], "weights_digest": man_p[model]["identity"]["weights_digest"],
                           "heads": list(POLICY_HEADS)},
                "value": {"tier": TIER_DIR[value_tier], "header_sha256": man_v[model]["header_sha256"],
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
    print(f"[export_mixed] {shown}: {len(manifest)} models, policy from {TIER_DIR[policy_tier]}, "
          f"value from {TIER_DIR[value_tier]}, buffer width {max(m['required_buffer_width'] for m in manifest.values())}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-tier", required=True, choices=sorted(TIER_DIR))
    ap.add_argument("--value-tier", required=True, choices=sorted(TIER_DIR))
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    export(a.policy_tier, a.value_tier, Path(a.out) if a.out else None)


if __name__ == "__main__":
    main()
