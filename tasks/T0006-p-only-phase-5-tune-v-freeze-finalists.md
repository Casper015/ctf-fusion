---
id: T0006
title: "P-only phase 5: freeze the finalists and confirm them"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/confirm/]
depends: [T0004, T0005, T0014]
land: pr
branch: ""
updated: 2026-10-11
---

## Goal

Confirm the frozen finalists: bots/p_only/TASK.md, Phase 5, 'Freeze, then confirm'.

Phase 5 also asks to compare V off, a simple score/time baseline and the `_m_v` hybrid at fixed P and to tune on tuning maps. That is the same work as T0014 steps 3 and 4 (screen at fixed P: V off, existing V, new small V, direct 4XL, distilled; PONLY_V_ROLLOUT tuned on its tuning seed; carrier, sizes, coefficients and budgets frozen), so it is done there, not here. Finalists may also come from T0013 (P family p3), T0019 (search portfolio) and T0022 (fresh-map V3-XL).

Here: take the frozen finalists (carrier, sizes, depth, coefficients, budgets) and confirm each on an unseen confirmation seed, chosen only after freezing.

## Done when

Confirmation with at least 10,000 independent map groups over counts 2-15 (at least 500 per count), three independently trained seeds and paired fixtures, reported under `bots/p_only/report/confirm/`.

## Handoff

Not started. Choose the confirmation seed only when the finalists are frozen, and keep it unseen until then.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-10 19:25Z claude@caspers-macbook: note: Candidates for this confirmation may come from T0013 (P feature family p3, equal-time screen) and T0014 (V pilot at fixed XXL P: retrained v2/v3 M, direct 4XL with the Route A bound, distilled). Their finalists arrive frozen (carrier, sizes, coefficients, budgets), with three independently trained seeds where a trained V or P is involved; docs/FEATURE_FAMILIES.md section 8.
- 2026-10-11 01:32Z claude-ci@caspers-macbook: amended depends -> T0004, T0005, T0014; title -> P-only phase 5: freeze the finalists and confirm them; goal
