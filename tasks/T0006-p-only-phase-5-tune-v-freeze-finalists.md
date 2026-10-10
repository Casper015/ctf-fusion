---
id: T0006
title: "P-only phase 5: tune V, freeze finalists, confirm"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/confirm/]
depends: [T0004, T0005]
land: pr
branch: ""
updated: 2026-10-10
---

## Goal

After P is settled: compare V off, a simple score/time baseline and the `_m_v` hybrid at fixed P. Tune on tuning maps only. Then freeze the finalists and confirm them on the unseen confirmation seed. Spec: [bots/p_only/TASK.md](../bots/p_only/TASK.md), Phase 5.

## Done when

Confirmation with at least 10,000 independent map groups over counts 2-15 (at least 500 per count), three independently trained seeds and paired fixtures, reported under `bots/p_only/report/confirm/`.

## Handoff

Not started. Choose the confirmation seed only when the finalists are frozen, and keep it unseen until then.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-10 19:25Z claude@caspers-macbook: note: Candidates for this confirmation may come from T0013 (P feature family p3, equal-time screen) and T0014 (V pilot at fixed XXL P: retrained v2/v3 M, direct 4XL with the Route A bound, distilled). Their finalists arrive frozen (carrier, sizes, coefficients, budgets), with three independently trained seeds where a trained V or P is involved; docs/FEATURE_FAMILIES.md section 8.
