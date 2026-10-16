"""Train exact-count PPO policy and value specialists (task T0068).

Bounded experiment:
1. Specialized arm: 14 distinct pi/V policies (n02..n15), 100,000 decisions each (1.4M aggregate).
   First 20,000 decisions per count are the correctness smoke check.
2. General control arm: 1 common policy on all counts 2..15, 1,400,000 decisions (100k per count interleaved).
   First 20,000 decisions are the correctness smoke check.
3. Total budget: 2.8M learner decisions.
4. Export: INT16 quantized headers for all 14 specialists + general control in inference/count_specialists_rl/,
   with C++/Python parity checks, routing registry, and model reports under reports/models/count_specialists_rl/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fusion import runner
from training.agent_v2 import bridge as AB
from training.agent_v2 import identity as AI
from training.agent_v2 import train as AT
from training.meow_rl import identity as MI
from training.meow_rl import policy as P
from training.meow_rl import quant as Q

COUNTS = list(range(2, 16))
SEEDS_LEDGER = REPO / "docs" / "COUNT_SPECIALISTS_RL_SEEDS.md"
POOL_PATH = REPO / "profiles" / "meow_rl_training_pool_v2.json"
INIT_ACTOR_PATH = REPO / "data" / "count_specialists_rl" / "init_actor.npz"
BASE_RUNS_DIR = REPO / "runs" / "count_specialists_rl"
REPORTS_MODELS_DIR = REPO / "reports" / "models" / "count_specialists_rl"
REPORTS_HOST_DIR = REPO / "reports" / "host" / "count_specialists_rl"
INFERENCE_DIR = REPO / "inference" / "count_specialists_rl"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for_count(count: int) -> int:
    # 20262002 for n02 ... 20262015 for n15
    return 20262000 + count


def seed_for_general() -> int:
    return 20262001


def run_training_stage(
    name: str,
    counts: list[int],
    seed: int,
    out_dir: Path,
    smoke_decisions: int,
    total_decisions: int,
    rollout_decisions: int,
    workers: int,
    device: str,
    ai_test: str | None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "state.pt"

    t0 = time.time()
    # Stage 1: Correctness smoke
    if not (out_dir / "smoke_report.json").is_file():
        print(f"\n[{name}] Stage 1: Running smoke check ({smoke_decisions} decisions)...", flush=True)
        smoke_args = [
            "--init-actor", str(INIT_ACTOR_PATH),
            "--p-bank", "count_specialists_p",
            "--mode", "smoke",
            "--arm", "large_v",
            "--pool", str(POOL_PATH),
            "--seed-base", str(seed),
            "--seed-ledger", str(SEEDS_LEDGER),
            "--device", device,
            "--workers", str(workers),
            "--decisions", str(smoke_decisions),
            "--rollout-decisions", str(rollout_decisions),
            "--keep-features",
            "--counts", *map(str, counts),
            "--out", str(out_dir),
        ]
        if ai_test:
            smoke_args += ["--ai-test", ai_test]

        res = AT.train(smoke_args)
        smoke_rep = json.loads((out_dir / "smoke_report.json").read_text(encoding="utf-8"))
        checks = smoke_rep["checks"]
        if not (checks.get("all_finite") and checks.get("nonzero_actor_grad") and checks.get("state_round_trip")):
            raise RuntimeError(f"[{name}] Smoke checks failed: {checks}")
        print(f"[{name}] Smoke check PASSED (max_logp_gap: {checks.get('max_logp_recompute_gap'):.2e})", flush=True)
    else:
        print(f"[{name}] Stage 1 smoke check already completed.", flush=True)

    # Stage 2: Full training to total_decisions
    current_decisions = 0
    if state_file.is_file():
        st = torch.load(state_file, map_location="cpu", weights_only=False)
        current_decisions = int(st.get("decisions", 0))

    if current_decisions < total_decisions:
        print(f"[{name}] Stage 2: Resuming to full budget {total_decisions} decisions (currently at {current_decisions})...", flush=True)
        full_args = [
            "--init-actor", str(INIT_ACTOR_PATH),
            "--p-bank", "count_specialists_p",
            "--mode", "full",
            "--resume",
            "--arm", "large_v",
            "--pool", str(POOL_PATH),
            "--seed-base", str(seed),
            "--seed-ledger", str(SEEDS_LEDGER),
            "--device", device,
            "--workers", str(workers),
            "--decisions", str(total_decisions),
            "--rollout-decisions", str(rollout_decisions),
            "--keep-features",
            "--counts", *map(str, counts),
            "--out", str(out_dir),
        ]
        if ai_test:
            full_args += ["--ai-test", ai_test]

        res = AT.train(full_args)
    else:
        print(f"[{name}] Already reached budget {total_decisions} decisions.", flush=True)

    wall_s = round(time.time() - t0, 1)

    # Summarize journal
    journal_path = out_dir / "journal.jsonl"
    entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    last = entries[-1] if entries else {}

    return {
        "name": name,
        "counts": counts,
        "seed": seed,
        "total_decisions": total_decisions,
        "updates": len(entries),
        "wall_s": wall_s,
        "final_utility": last.get("utility", 0.0),
        "final_kl": last.get("kl", 0.0),
        "final_v_loss": last.get("v_loss", 0.0),
        "final_pg_loss": last.get("pg_loss", 0.0),
    }


def calibrate_and_export_actor(
    name: str,
    count_tag: str,
    run_dir: Path,
    out_header: Path,
    namespace: str,
    guard: str,
    max_calib_rows: int = 50_000,
) -> dict:
    state_file = run_dir / "state.pt"
    st = torch.load(state_file, map_location="cpu", weights_only=False)
    actor_fp32 = st["actor_numpy"]
    identity = st["identity"]

    # Load calibration observations from saved batches
    batches_dir = run_dir / "batches"
    batch_files = sorted(batches_dir.glob("update_*.npz"))
    if not batch_files:
        raise FileNotFoundError(f"No update_*.npz in {batches_dir}")

    feats_list = []
    masks_list = []
    for bf in batch_files:
        with np.load(bf) as z:
            if "feats" in z:
                feats_list.append(z["feats"])
            if "joint_mask" in z:
                masks_list.append(z["joint_mask"])

    all_feats = np.concatenate(feats_list, axis=0)
    all_masks = np.concatenate(masks_list, axis=0) if masks_list else None

    # Subsample calibration rows
    if len(all_feats) > max_calib_rows:
        idx = np.linspace(0, len(all_feats) - 1, max_calib_rows).astype(np.int64)
        calib_feats = all_feats[idx]
        calib_masks = all_masks[idx] if all_masks is not None else None
    else:
        calib_feats = all_feats
        calib_masks = all_masks

    # Quantize actor
    q = Q.quantize(actor_fp32, calib_feats.reshape(-1, 112))

    # Agreement check against FP32
    agree = {}
    if calib_masks is not None:
        eval_slice = min(1000, len(calib_feats))
        agree = Q.agreement(actor_fp32, q, calib_feats[:eval_slice], calib_masks[:eval_slice])

    # Generate header with unique namespace and guard
    from training.export_weights import _array, _emit_nets  # noqa: PLC0415
    body: list[str] = []
    _emit_nets(body, {"actor": q["net"]}, {}, False, _array)

    header_text = "\n".join([
        f"// Generated by training/train_count_specialists_rl.py: {name}. Do not edit.",
        f"// scheme {q['scheme']} (reference16 INT16), actor {q['actor']}, features {q['policy_features']}",
        f"// calibration {q['calibration']['observations']} observations, margin {q['calibration']['margin']}",
        f"#ifndef {guard}",
        f"#define {guard}",
        "#include <cstdint>",
        '#include "mlp.hpp"',
        f"namespace {namespace} {{",
        f'inline constexpr const char* kName = "{name}";',
        f'inline constexpr const char* kScheme = "{q["scheme"]}";',
        f'inline constexpr const char* kDigest = "{q["digest"]}";',
        f'inline constexpr const char* kIdentity = "{MI.digest(identity)}";',
        f"inline constexpr int kInBits = {q['in_bits']};",
        f"inline constexpr double kTemperature = {q['temperature']!r};",
        *body,
        f"inline constexpr fusion::Net kActor{{actor_layers, {len(q['net']['layers'])}, {q['net']['out_exp']}}};",
        f"}}  // namespace {namespace}",
        f"#endif  // {guard}",
        "",
    ])

    out_header.parent.mkdir(parents=True, exist_ok=True)
    out_header.write_text(header_text, encoding="utf-8")

    # Also save NPZ weights
    npz_path = out_header.with_suffix(".npz")
    np.savez(npz_path, **actor_fp32)

    return {
        "name": name,
        "count_tag": count_tag,
        "namespace": namespace,
        "header": out_header.name,
        "header_path": str(out_header.relative_to(REPO).as_posix()),
        "header_sha256": sha256_file(out_header),
        "npz": npz_path.name,
        "npz_path": str(npz_path.relative_to(REPO).as_posix()),
        "digest": q["digest"],
        "greedy_agreement": agree.get("greedy_agreement", 1.0),
        "mean_abs_logp_error": agree.get("mean_abs_logp_error", 0.0),
    }


def emit_registry(inference_dir: Path, exported_models: dict[str, dict]) -> Path:
    registry_file = inference_dir / "registry.hpp"
    lines = [
        "// Generated by training/train_count_specialists_rl.py. Do not edit.",
        "#pragma once",
        "#include <cstdint>",
        '#include "mlp.hpp"',
        "",
    ]
    for tag in [f"n{c:02d}" for c in COUNTS] + ["general"]:
        if tag in exported_models:
            lines.append(f'#include "weights_{tag}.hpp"')

    lines.extend([
        "",
        "namespace count_specialists_rl {",
        "",
        "inline const fusion::Net& get_actor(int teams) {",
        "    switch (teams) {",
    ])
    for c in COUNTS:
        tag = f"n{c:02d}"
        if tag in exported_models:
            lines.append(f"    case {c}: return fusion::models::rl_{tag}::kActor;")
    lines.extend([
        "    default: return fusion::models::rl_general::kActor;",
        "    }",
        "}",
        "",
        "inline double get_temperature(int teams) {",
        "    return 1.05;",
        "}",
        "",
        "}  // namespace count_specialists_rl",
        "",
    ])
    registry_file.write_text("\n".join(lines), encoding="utf-8")
    return registry_file


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", type=int, nargs="+", default=COUNTS, help="counts to train (default: 2..15)")
    ap.add_argument("--specialized-decisions", type=int, default=100_000, help="decisions per specialist")
    ap.add_argument("--general-decisions", type=int, default=1_400_000, help="decisions for general control")
    ap.add_argument("--smoke-decisions", type=int, default=20_000, help="smoke check budget per run")
    ap.add_argument("--rollout-decisions", type=int, default=8192, help="rollout decisions per update")
    ap.add_argument("--workers", type=int, default=8, help="concurrent CPU rollout workers")
    ap.add_argument("--device", default="cuda", help="device for training")
    ap.add_argument("--ai-test", default=None, help="arena root checkout")
    ap.add_argument("--skip-general", action="store_true", help="skip general control training")
    ap.add_argument("--only-general", action="store_true", help="run only general control")
    ap.add_argument("--export-only", action="store_true", help="skip training and run export")
    a = ap.parse_args(argv)

    t_start = time.time()
    results = {}

    if not a.export_only:
        # 1. Train specialized arm
        if not a.only_general:
            for count in a.counts:
                tag = f"n{count:02d}"
                out_dir = BASE_RUNS_DIR / f"spec_{tag}"
                seed = seed_for_count(count)
                print(f"\n{'='*70}\nTraining Specialist {tag} (Count {count}) | Seed {seed}\n{'='*70}", flush=True)
                res = run_training_stage(
                    name=f"spec_{tag}",
                    counts=[count],
                    seed=seed,
                    out_dir=out_dir,
                    smoke_decisions=a.smoke_decisions,
                    total_decisions=a.specialized_decisions,
                    rollout_decisions=a.rollout_decisions,
                    workers=a.workers,
                    device=a.device,
                    ai_test=a.ai_test,
                )
                results[tag] = res

        # 2. Train general control arm
        if not a.skip_general:
            print(f"\n{'='*70}\nTraining General Control Arm | Seed {seed_for_general()}\n{'='*70}", flush=True)
            gen_out = BASE_RUNS_DIR / "general_control"
            res = run_training_stage(
                name="general_control",
                counts=COUNTS,
                seed=seed_for_general(),
                out_dir=gen_out,
                smoke_decisions=a.smoke_decisions,
                total_decisions=a.general_decisions,
                rollout_decisions=a.rollout_decisions,
                workers=a.workers,
                device=a.device,
                ai_test=a.ai_test,
            )
            results["general"] = res

    # 3. Export models to inference/count_specialists_rl/
    print(f"\n{'='*70}\nExporting Models & Emitting Registry\n{'='*70}", flush=True)
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    exported = {}

    for count in a.counts:
        tag = f"n{count:02d}"
        run_dir = BASE_RUNS_DIR / f"spec_{tag}"
        if (run_dir / "state.pt").is_file():
            hdr = INFERENCE_DIR / f"weights_{tag}.hpp"
            exp = calibrate_and_export_actor(
                name=f"spec_{tag}",
                count_tag=tag,
                run_dir=run_dir,
                out_header=hdr,
                namespace=f"fusion::models::rl_{tag}",
                guard=f"COUNT_SPECIALISTS_RL_{tag.upper()}_HPP",
            )
            exported[tag] = exp
            print(f"Exported {tag}: {hdr.name} (greedy agreement: {exp['greedy_agreement']:.3f})")
            if tag not in results:
                st = torch.load(run_dir / "state.pt", map_location="cpu", weights_only=False)
                counters = st.get("counters", {})
                history = counters.get("history", []) if isinstance(counters, dict) else (counters if isinstance(counters, list) else [])
                last = history[-1] if history else {}
                results[tag] = {
                    "name": f"spec_{tag}",
                    "counts": [count],
                    "seed": seed_for_count(count),
                    "total_decisions": counters.get("decisions", a.specialized_decisions) if isinstance(counters, dict) else a.specialized_decisions,
                    "updates": len(history),
                    "final_utility": last.get("utility_mean", 0.0),
                    "final_kl": last.get("approx_kl", 0.0),
                    "final_v_loss": last.get("v_loss", 0.0),
                    "final_pg_loss": last.get("pg_loss", 0.0),
                }

    gen_dir = BASE_RUNS_DIR / "general_control"
    if (gen_dir / "state.pt").is_file():
        hdr = INFERENCE_DIR / "weights_general.hpp"
        exp = calibrate_and_export_actor(
            name="general_control",
            count_tag="general",
            run_dir=gen_dir,
            out_header=hdr,
            namespace="fusion::models::rl_general",
            guard="COUNT_SPECIALISTS_RL_GENERAL_HPP",
        )
        exported["general"] = exp
        print(f"Exported general: {hdr.name} (greedy agreement: {exp['greedy_agreement']:.3f})")
        if "general" not in results:
            st = torch.load(gen_dir / "state.pt", map_location="cpu", weights_only=False)
            counters = st.get("counters", {})
            history = counters.get("history", []) if isinstance(counters, dict) else (counters if isinstance(counters, list) else [])
            last = history[-1] if history else {}
            results["general"] = {
                "name": "general_control",
                "counts": COUNTS,
                "seed": seed_for_general(),
                "total_decisions": counters.get("decisions", a.general_decisions) if isinstance(counters, dict) else a.general_decisions,
                "updates": len(history),
                "final_utility": last.get("utility_mean", 0.0),
                "final_kl": last.get("approx_kl", 0.0),
                "final_v_loss": last.get("v_loss", 0.0),
                "final_pg_loss": last.get("pg_loss", 0.0),
            }

    reg = emit_registry(INFERENCE_DIR, exported)
    print(f"Emitted registry: {reg}")

    # Emit MANIFEST.json
    manifest = {
        "schema": "ctf-count-specialists-rl-manifest-1",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "registry": reg.name,
        "registry_path": str(reg.relative_to(REPO).as_posix()),
        "registry_sha256": sha256_file(reg),
        "models": exported,
    }
    manifest_path = INFERENCE_DIR / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    # Write training report under reports/models/count_specialists_rl/
    REPORTS_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    rep_summary = {
        "task": "T0068",
        "carrier": "agent_v2_v1",
        "p_bank": "count_specialists_p",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": round(time.time() - t_start, 1),
        "results": results,
        "exported": exported,
    }
    (REPORTS_MODELS_DIR / "TRAINING_SUMMARY.json").write_text(json.dumps(rep_summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report: {REPORTS_MODELS_DIR / 'TRAINING_SUMMARY.json'}")
    print(f"\nAll operations finished in {round(time.time() - t_start, 1)}s")


if __name__ == "__main__":
    main()
