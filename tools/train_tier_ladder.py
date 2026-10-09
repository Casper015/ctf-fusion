"""Train, evaluate, and export model matrices for size tiers (M, L, XL, XXL).

Usage:
    python tools/train_tier_ladder.py --tiers M
    python tools/train_tier_ladder.py --tiers M L XL XXL
    python tools/train_tier_ladder.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TIERS = {
    "M": {
        "mover": "32",
        "squad": "128,64",
        "act": "32",
        "target": "16",
        "value": "256,32",
        "max_width": 256,
    },
    "L": {
        "mover": "64",
        "squad": "256,64",
        "act": "64",
        "target": "16",
        "value": "512,64",
        "max_width": 512,
    },
    "XL": {
        "mover": "128",
        "squad": "512,128",
        "act": "128",
        "target": "32",
        "value": "1024,64",
        "max_width": 1024,
    },
    "XXL": {
        "mover": "256",
        "squad": "1024,256",
        "act": "256",
        "target": "32",
        "value": "2048,128",
        "max_width": 2048,
    },
}

DEFAULT_MODELS = [
    "general",
    "n02", "n03", "n04", "n05", "n06",
    "n07", "n08", "n10", "n12", "n15",
]


def run_cmd(cmd: list[str], dry_run: bool = False) -> None:
    print(f"\n[RUN] {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(REPO))
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[FAIL] command returned {res.returncode} after {dt:.1f}s", flush=True)
        sys.exit(res.returncode)
    print(f"[DONE] finished in {dt:.1f}s", flush=True)


def is_run_complete(run_dir: Path) -> bool:
    return (run_dir / "model.npz").exists() and (run_dir / "config.json").exists()


def is_eval_complete(report_dir: Path, model: str) -> bool:
    return (report_dir / f"{model}.json").exists() and (report_dir / f"{model}.md").exists()


def is_export_complete(export_dir: Path, model: str) -> bool:
    hpp = export_dir / f"weights_{model}.hpp"
    npz = export_dir / f"weights_{model}.npz"
    manifest = export_dir / "MANIFEST.json"
    if not (hpp.exists() and npz.exists() and manifest.exists()):
        return False
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        return model in data
    except Exception:
        return False


def train_tier(
    tier: str,
    models: list[str],
    shards: str,
    epochs: int,
    skip_completed: bool,
    dry_run: bool,
) -> None:
    if tier not in TIERS:
        raise ValueError(f"Unknown tier: {tier}. Available: {list(TIERS.keys())}")

    cfg = TIERS[tier]
    tier_lower = tier.lower()
    runs_dir = REPO / f"runs_{tier_lower}"
    reports_dir = REPO / f"reports_{tier_lower}"
    models_dir = REPO / "inference" / f"models_{tier_lower}"

    runs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\nStarting Size Tier {tier} (max_width={cfg['max_width']})\n{'='*70}", flush=True)

    # 1. Train Value models
    for m in models:
        out_v = runs_dir / f"value_{m}"
        if skip_completed and is_run_complete(out_v):
            print(f"[SKIP] Value {tier}:{m} already complete at {out_v}", flush=True)
            continue

        cmd = [
            sys.executable, "-m", "training.train_value",
            "--shards", shards,
            "--model", m,
            "--hidden", cfg["value"],
            "--max-width", str(cfg["max_width"]),
            "--epochs", str(epochs),
            "--out", str(out_v),
        ]
        if m == "general":
            cmd += ["--count-weighting", "inv_teams"]
        else:
            cmd += ["--init", str(runs_dir / "value_general")]
        run_cmd(cmd, dry_run=dry_run)

    # 2. Train Policy models
    for m in models:
        out_p = runs_dir / f"policy_{m}"
        if skip_completed and is_run_complete(out_p):
            print(f"[SKIP] Policy {tier}:{m} already complete at {out_p}", flush=True)
            continue

        cmd = [
            sys.executable, "-m", "training.train_policy",
            "--shards", shards,
            "--model", m,
            "--hidden-mover", cfg["mover"],
            "--hidden-squad", cfg["squad"],
            "--hidden-act", cfg["act"],
            "--hidden-target", cfg["target"],
            "--max-width", str(cfg["max_width"]),
            "--epochs", str(epochs),
            "--out", str(out_p),
        ]
        if m == "general":
            cmd += ["--count-weighting", "inv_teams"]
        else:
            cmd += ["--init", str(runs_dir / "policy_general")]
        run_cmd(cmd, dry_run=dry_run)

    # 3. Offline Evaluation
    for m in models:
        if skip_completed and is_eval_complete(reports_dir, m):
            print(f"[SKIP] Eval {tier}:{m} already complete at {reports_dir}/{m}.json", flush=True)
            continue

        cmd = [
            sys.executable, "-m", "training.evaluate_offline",
            "--shards", shards,
            "--model", m,
            "--policy", str(runs_dir / f"policy_{m}"),
            "--value", str(runs_dir / f"value_{m}"),
            "--max-width", str(cfg["max_width"]),
            "--out", str(reports_dir / m),
        ]
        run_cmd(cmd, dry_run=dry_run)

    # 4. Export Weights
    for m in models:
        if skip_completed and is_export_complete(models_dir, m):
            print(f"[SKIP] Export {tier}:{m} already complete in {models_dir}", flush=True)
            continue

        report_file = reports_dir / f"{m}.json"
        cmd = [
            sys.executable, "-m", "training.export_weights",
            "--model", m,
            "--policy", str(runs_dir / f"policy_{m}"),
            "--value", str(runs_dir / f"value_{m}"),
            "--report", str(report_file),
            "--max-width", str(cfg["max_width"]),
            "--out", str(models_dir),
        ]
        run_cmd(cmd, dry_run=dry_run)

    # 5. Summarize Tier Results
    if not dry_run:
        manifest_file = models_dir / "MANIFEST.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            summary_lines = [
                f"# Size Tier {tier} Export Summary",
                f"- Models exported: {len(manifest)} / {len(models)}",
                f"- Required FUSION_MAX_WIDTH: {cfg['max_width']}",
                "",
                "| Model | Params | Mover | Squad | Act | Tgt | Val | Per-Team MACs | Gated Alphas |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
            for name, item in manifest.items():
                p = item["params"]
                total_p = sum(p.values())
                alphas = ", ".join(f"{t}: {a:.3f}" for t, a in sorted(item.get("alpha", {}).items())) or "none"
                macs = item.get("macs", {}).get("joint_policy_per_team", 0)
                summary_lines.append(
                    f"| `{name}` | {total_p:,} | {p['mover']} | {p['squad']} | {p['act']} | {p['target']} | {p['value']} | {macs:,} | {alphas} |"
                )
            summary_path = reports_dir / "SUMMARY.md"
            summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
            print(f"\n[SUMMARY] Wrote tier {tier} summary to {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train size ladder matrices (M, L, XL, XXL)")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS.keys()), default=["M", "L", "XL", "XXL"])
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--shards", default="data/shards")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--no-skip", action="store_true", help="Do not skip completed artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without running")
    args = parser.parse_args()

    for tier in args.tiers:
        train_tier(
            tier=tier,
            models=args.models,
            shards=args.shards,
            epochs=args.epochs,
            skip_completed=not args.no_skip,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
