---
id: T0025
title: "P on unseen opponents: offline check on the two reserved lineages"
status: open
owner: ""
needs: [host]
writes: [reports/models/p_ood/, tests/test_repo_layout.py]
depends: []
land: direct
branch: ""
updated: 2026-10-11
---

## Goal

No report measures P on opponents it never trained against. Every gate report has `holdout_family=[]`; the two lineages the pool reserves for that purpose, AI_Arena_CTF_Player02_bot and ctf-arena_Player4_v2 ([training_selection_20261009](../roster/report/training_selection_20261009/README.md)), are outside collection, so no data contains them ([MODEL_EXPORT_REVIEW_8595c36.md](../reports/reviews/MODEL_EXPORT_REVIEW_8595c36.md), section on OOD). Collect fresh-map games under official limits in which those lineages play as opponents (a new seed, not one in the seed ledger of docs/DIGEST.md), then evaluate the deployed P (XL and XXL, family p2) on their rows only with `training.evaluate_offline`: joint-move log-loss against q_H, inertia and uniform, calibration, per count, with map-clustered intervals, next to the same models' in-distribution test numbers. From the triage of the pre-board prompts (T0024): diverse15 Phase A step 8 and Phase E, review bundle Phase 4.

## Done when

`reports/models/p_ood/` holds the offline report for XL and XXL (per count; reserved lineages only; the in-distribution numbers alongside) and states the seed and the collection command. `p_ood` is a study folder in tests/test_repo_layout.py MODEL_STUDIES; tests pass.

## Handoff

Not started.

## Log

- 2026-10-11 01:51Z claude-ci@caspers-macbook: created
