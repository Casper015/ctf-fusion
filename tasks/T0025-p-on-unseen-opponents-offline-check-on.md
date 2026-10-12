---
id: T0025
title: "P on unseen opponents: offline check on the two reserved lineages"
status: done
owner: codex-review@caspers-macbook
needs: [host]
writes: [reports/models/p_ood/, tests/test_repo_layout.py]
depends: []
land: direct
branch: codex-review/t0025-p-on-unseen-opponents-offline-check-on
updated: 2026-10-12
---

## Goal

No report measures P on opponents it never trained against. Every gate report has `holdout_family=[]`; the two lineages the pool reserves for that purpose, AI_Arena_CTF_Player02_bot and ctf-arena_Player4_v2 ([training_selection_20261009](../roster/report/training_selection_20261009/README.md)), are outside collection, so no data contains them ([MODEL_EXPORT_REVIEW_8595c36.md](../reports/reviews/MODEL_EXPORT_REVIEW_8595c36.md), section on OOD). Collect fresh-map games under official limits in which those lineages play as opponents (a new seed, not one in the seed ledger of docs/DIGEST.md), then evaluate the deployed P (XL and XXL, family p2) on their rows only with `training.evaluate_offline`: joint-move log-loss against q_H, inertia and uniform, calibration, per count, with map-clustered intervals, next to the same models' in-distribution test numbers. From the triage of the pre-board prompts (T0024): diverse15 Phase A step 8 and Phase E, review bundle Phase 4.

## Done when

`reports/models/p_ood/` holds the offline report for XL and XXL (per count; reserved lineages only; the in-distribution numbers alongside) and states the seed and the collection command. `p_ood` is a study folder in tests/test_repo_layout.py MODEL_STUDIES; tests pass.

## Handoff

*2026-10-12 16:09Z codex-review@caspers-macbook:*

Accepted and verified merged: PR #27 (9350d8ca9afc81932de36056abf3a025cf98d406). Review https://github.com/Casper015/ctf-fusion/pull/27#pullrequestreview-5493228470. Report in reports/models/p_ood/: 300 maps, 224000 valid reserved-lineage rows; prediction metrics only. Exact-head CI succeeds, local report-layout checks pass. Existing native/168-cohort audits inspected; no successful measured stage rerun. Closing stale review status under Casper's explicit authorization to approve current PRs and reconcile tasks.

## Log

- 2026-10-11 01:51Z claude-ci@caspers-macbook: created
- 2026-10-12 05:20Z codex@desktop-g60ojud: handoff -> review
- 2026-10-12 16:08Z codex-review@caspers-macbook: note: Casper explicitly authorized acceptance/status reconciliation; reviewed merged deliverable before reviewer takeover.
- 2026-10-12 16:08Z codex-review@caspers-macbook: claimed (forced)
- 2026-10-12 16:09Z codex-review@caspers-macbook: done
