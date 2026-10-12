---
id: T0025
title: "P on unseen opponents: offline check on the two reserved lineages"
status: review
owner: codex@desktop-g60ojud
needs: [host]
writes: [reports/models/p_ood/, tests/test_repo_layout.py]
depends: []
land: direct
branch: ""
updated: 2026-10-12
---

## Goal

No report measures P on opponents it never trained against. Every gate report has `holdout_family=[]`; the two lineages the pool reserves for that purpose, AI_Arena_CTF_Player02_bot and ctf-arena_Player4_v2 ([training_selection_20261009](../roster/report/training_selection_20261009/README.md)), are outside collection, so no data contains them ([MODEL_EXPORT_REVIEW_8595c36.md](../reports/reviews/MODEL_EXPORT_REVIEW_8595c36.md), section on OOD). Collect fresh-map games under official limits in which those lineages play as opponents (a new seed, not one in the seed ledger of docs/DIGEST.md), then evaluate the deployed P (XL and XXL, family p2) on their rows only with `training.evaluate_offline`: joint-move log-loss against q_H, inertia and uniform, calibration, per count, with map-clustered intervals, next to the same models' in-distribution test numbers. From the triage of the pre-board prompts (T0024): diverse15 Phase A step 8 and Phase E, review bundle Phase 4.

## Done when

`reports/models/p_ood/` holds the offline report for XL and XXL (per count; reserved lineages only; the in-distribution numbers alongside) and states the seed and the collection command. `p_ood` is a study folder in tests/test_repo_layout.py MODEL_STUDIES; tests pass.

## Handoff

*2026-10-12 05:20Z codex@desktop-g60ojud:*

Ready for review: PR #27 https://github.com/Casper015/ctf-fusion/pull/27; head 357eb252e3810054cc4b5f3e162dd0ce0e3ca082, branch codex/t0025-unseen-lineage-evidence. Frozen measurement source de5dece1b90206684e06d82617d772e82d6658a3, consumed seed 20261250: 300 official games, 224000 valid reserved-lineage rows, zero all-seat faults. Full source/training-lineage/raw audits, 28 native/actual-NPZ parity cases, pure-P/qH/inertia/uniform offline scoring and 168-cohort independent metric audit complete. XL/XXL NLL 2.113965/2.103551; XXL minus XL -0.010414 nats CI[-0.012779,-0.008049], not a playing-strength result. Reports/models/p_ood/RESULTS.md, CHECKS.md and reproduction/receipts committed. All 224 core tests covered: 223 pass, one platform skip; six layout checks pass. T0030 now done releases registration lane; only p_ood registration added. Exact-head CI pending. No replay/training/export/default change; do not restart successful stages. User authorized status bookkeeping; keep review until accepted/merged.

## Log

- 2026-10-11 01:51Z claude-ci@caspers-macbook: created
- 2026-10-12 05:20Z codex@desktop-g60ojud: handoff -> review
- 2026-10-12 16:08Z codex-review@caspers-macbook: note: Casper explicitly authorized acceptance/status reconciliation; reviewed merged deliverable before reviewer takeover.
