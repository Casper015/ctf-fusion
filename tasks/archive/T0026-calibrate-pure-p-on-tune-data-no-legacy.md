---
id: T0026
title: "Calibrate pure P on tune data (no legacy blend)"
status: dropped
owner: ""
needs: [cuda]
writes: [reports/models/p_calibration/, tests/test_repo_layout.py]
depends: []
land: direct
branch: ""
updated: 2026-10-18
---

## Goal

[bots/p_only/README.md](../../bots/p_only/README.md) says calibration of pure P on tune data is open: the P-only bots use P with no legacy blend (the gates gave P 96-99% weight in the mixture), but each exported temperature was fitted for the gate. Measure pure P's calibration (NLL, ECE, top-1, per count) for the deployed XL and XXL P on the fit split, refit the temperature for pure P where it helps, and report the change. A re-export changes the model identity that every P-only screen binds to, so export nothing until T0006's finalists are chosen; record the fitted temperatures instead. From the triage of the pre-board prompts (T0024): review bundle Phase 1 step 3, diverse15 Phase E ('fit temperatures on tune, freeze them').

## Done when

`reports/models/p_calibration/` holds the per-count report (before and after the refit, map-clustered intervals) and the fitted temperatures; no exported artifact changed. `p_calibration` is a study folder in tests/test_repo_layout.py MODEL_STUDIES; tests pass.

## Handoff

*2026-10-18 19:37Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). GPU-only backlog not on the current plan; any GPU run must also stay out of host timing windows. Reopen as a new task if needed.

## Log

- 2026-10-11 01:51Z claude-ci@caspers-macbook: created
- 2026-10-18 19:37Z claude-fervent@cloud: dropped
