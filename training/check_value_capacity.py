"""Verify an actual trained shared V export, frozen P copies and native full/incremental/Python parity.

python -m training.check_value_capacity --set build/v_capacity_5xl/sets/5XL \
    --run runs/v_capacity_5xl/5XL_s1 --cache data/vcache/designated_v3 \
    --out reports/models/v_capacity_5xl/integer_5XL.json

Compile before evidence runs. Checks and quantization errors use validation only;
this is correctness evidence, not a full-turn timing or model-quality approval.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fusion import native, quant
from fusion.capacity import buffer_width
from fusion.paths import REPO, sha256
from fusion.schema import layout
from training.export_mixed import POLICY_HEADS, head_arrays, quantize_run_value
from training.export_weights import load_value, macs
from training.header_chunks import verify_chunks
from training.value_cache import load_cache


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("set", "run", "cache", "out"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--rows", type=int, default=256)
    a = ap.parse_args(argv)
    root, L = Path(a.set), layout()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    q, feats = quantize_run_value(Path(a.run), L, 16384)
    value, beta = load_value(Path(a.run), L)
    reference = quant.flatten(q, "value")
    shared = manifest["general"]["shared_value"]
    header = root / shared["header"]
    verify_chunks(root, shared.get("chunks", []))
    if sha256(header) != shared["sha256"]:
        raise ValueError("shared value header differs from manifest")
    npz = header.with_suffix(".npz")
    if sha256(npz) != shared["npz_sha256"]:
        raise ValueError("shared value arrays differ from manifest")
    with np.load(npz) as saved:
        for k, v in reference.items():
            np.testing.assert_array_equal(saved[k], v)
    for name, entry in manifest.items():
        if entry["shared_value"] != shared:
            raise ValueError("team-count models do not share exactly one V")
        with np.load(root / f"weights_{name}.npz") as saved, np.load(REPO / f"inference/models_xxl/weights_{name}.npz") as frozen:
            pol = {k: v for h in POLICY_HEADS for k, v in head_arrays(dict(frozen), h).items()}
            if set(saved.files) != set(pol):
                raise ValueError("a model duplicated V or omitted frozen P arrays")
            for k, v in pol.items():
                if saved[k].dtype != v.dtype or saved[k].shape != v.shape:
                    raise ValueError(f"frozen P format changed: {name}/{k}")
                np.testing.assert_array_equal(saved[k], v)
    d = load_cache(a.cache, "val", feats["family"])
    indices = np.unique(np.linspace(0, len(d["v"]) - 1, min(a.rows, len(d["v"])), dtype=int))
    x = np.asarray(d["v"][indices]).astype("<i2")
    required = buffer_width(int(dim) for layer in q["layers"] for dim in layer["w"].shape)
    binary = native.build(REPO / "tools/cpp/value_capacity_check.cpp", "value_capacity_export", deps=native.model_dependencies(header),
                          defines=[f'FUSION_MODEL_HEADER="{header.resolve().as_posix()}"',
                                   "FUSION_MODEL_NS=fusion::models::shared_value", f"FUSION_MAX_WIDTH={required}"],
                          includes=[REPO / "inference"])
    result = native.run_tool(binary, x.tobytes())
    if result.returncode:
        raise ValueError(f"native value parity/load failed ({result.returncode}): {result.stderr.decode()}")
    raw = np.frombuffer(result.stdout, "<i4").reshape(len(x), 3)
    np.testing.assert_array_equal(raw[:, 0], raw[:, 1])
    np.testing.assert_array_equal(raw[:, 0], raw[:, 2])
    np.testing.assert_array_equal(raw[:16, :1], quant.forward_raw(q, x[:16]))
    floating = beta * value.logits((x / L.value_scale(feats["family"])).astype(np.float32))[:, 0].astype(np.float64)
    integer = quant.to_real(q, raw[:, 0])
    pf, pi = 1 / (1 + np.exp(-floating)), 1 / (1 + np.exp(-integer))
    report = {"run_checkpoint_sha256": sha256(Path(a.run) / "model.npz"), "manifest_sha256": sha256(root / "MANIFEST.json"),
              "binary_sha256": sha256(binary), "family": feats, "width": required, "training_parameters": sum(w.size + b.size for w, b in zip(value.net.W, value.net.b)),
              "outcome_macs": macs(q), "policy_models_copied_exactly": len(manifest), "shared_value_copies": 1,
              "layers": [{"shape": list(l["w"].shape), "shift": l["shift"], "exponent": l["exp"], "bound": l["bound"]} for l in q["layers"]],
              "validation_indices": indices.tolist(), "parity_rows": len(x), "python_parity_rows": min(16, len(x)),
              "full_incremental_row_major_transposed": "exact", "native_startup_and_embedded_load": "passed, not timing evidence",
              "quantization_error": {"max_logit": float(np.abs(integer - floating).max()), "mean_logit": float(np.abs(integer - floating).mean()),
                                     "max_probability": float(np.abs(pi - pf).max()), "mean_probability": float(np.abs(pi - pf).mean())},
              "header_bytes": header.stat().st_size, "chunk_bytes": sum(c["bytes"] for c in shared.get("chunks", [])),
              "largest_chunk_bytes": max((c["bytes"] for c in shared.get("chunks", [])), default=0), "chunks": len(shared.get("chunks", [])),
              "value_approved": False}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("training_parameters", "outcome_macs", "quantization_error", "chunks", "chunk_bytes")}, indent=1))


if __name__ == "__main__":
    main()
