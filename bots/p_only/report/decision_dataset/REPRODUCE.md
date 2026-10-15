# T0038 Decision Dataset and Quality Audit Reproduction

This document provides exact reproduction instructions for auditing the T0037 Windows host decision pilot corpus (seed `20261701`), building the authoritative decision training dataset (`ctf-fusion-decision-2`), and evaluating noise, coverage, and out-of-sample headroom diagnostics.

All successful pilot fixtures and continuations are immutable. Do not re-run matches or re-sample games.

## 1. Prerequisites & Environment

- **Host Machine:** Windows (`DESKTOP-G60OJUD`)
- **Python:** Python 3.10+ (or Python 3.14 on Windows)
- **Dependencies:** NumPy, PyTorch (CPU or CUDA)
- **Worktree:** `ctf-fusion.worktrees/T0038` on branch `codex-data-audit/t0038-audit-expanded-decision-pilot-and-build`
- **Raw Pilot Corpus:** Located at `data/decision/host_20261014/` (from T0037, seed 20261701)

## 2. Independent Corpus Audit

Run the independent audit script to verify all 336 matches, 336 branch roots, 5,268 continuations, and prefix causal feature extraction:

```powershell
$env:PYTHONPATH = "."
python bots/p_only/report/decision_dataset/audit_pilot.py
```

This verifies:
1. All 336 matches completed 400 turns with zero faults across all seats.
2. Carrier binary SHA256 matches `981a1e3c580021c6b5879e3ca3feb6de44642db4272aba25e5c97be22bac45b3`.
3. Candidate 0 matches the on-policy recorded reply.
4. Shortlist carrier scores are present and valid for all roots.
5. All 5,268 continuations have zero prefix, root-computed, and live faults.
6. Recomputed utilities from raw team score vectors match stored outcomes within $10^{-9}$.
7. Generates `bots/p_only/report/decision_dataset/DATA_AUDIT.json`.

## 3. Build Authoritative Decision Dataset

Execute `training.decision_dataset` with `--authoritative`:

```powershell
$env:PYTHONPATH = "."
python -m training.decision_dataset `
    --branches data/decision/host_20261014/branches `
    --matches data/decision/host_20261014/matches `
    --audit bots/p_only/report/decision_dataset/DATA_AUDIT.json `
    --manifest-receipt bots/p_only/report/decision_dataset/MANIFEST_METADATA_ADAPTER.json `
    --carrier-config bots/p_only/report/decision_dataset/carrier_config.json `
    --accept-selected-scope `
    --authoritative `
    --out data/decision/host_pilot_dataset
```

Output:
- `data/decision/host_pilot_dataset/train/decision.npz` (268 roots, 1,048 candidates)
- `data/decision/host_pilot_dataset/val/decision.npz` (37 roots, 148 candidates)
- `data/decision/host_pilot_dataset/test/decision.npz` (31 roots, 121 candidates)
- `data/decision/host_pilot_dataset/index.json` (authoritative: true)

## 4. Run Coverage, Noise and Power Diagnostics

Run `training.decision_noise` on the generated dataset:

```powershell
$env:PYTHONPATH = "."
python -m training.decision_noise `
    --data data/decision/host_pilot_dataset `
    --out reports/eval/decision_data_quality/noise
```

Outputs:
- `reports/eval/decision_data_quality/noise.json`
- `reports/eval/decision_data_quality/noise.md`

## 5. Artifact Hashes & Verified Digests

- `DATA_AUDIT.json`: SHA256 verified in `CHECKS.json`
- `MANIFEST_METADATA_ADAPTER.json`: `verification_scope: selected_frozen_identities_only`
- `carrier_config.json`: Validated against `inference/models_xxl/weights_general.hpp`
- `DATA_QUALITY_REPORT.md`: Comprehensive audit report and GO recommendation
