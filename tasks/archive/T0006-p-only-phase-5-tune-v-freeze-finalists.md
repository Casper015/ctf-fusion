---
id: T0006
title: "P-only phase 5: freeze the finalists and confirm them"
status: dropped
owner: ""
needs: [host]
writes: [bots/p_only/report/confirm/]
depends: [T0004, T0005, T0013]
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

Confirm the frozen admitted P/search/decision finalists on an unseen confirmation seed, chosen only after freezing.

Casper retired the unstarted V pilot T0012 and T0014 on 2026-10-11 (America/Los_Angeles). They are not dependencies and this task does not revive supervised-V training, distillation or leaf-coefficient tuning. Keep V off; retain completed V reports as historical evidence.

Candidates may come from T0013 (original P2/P3 equal-time screen), T0005 (four-turn search) and, only after separate CPU export/runtime/equal-time admission, T0032's distilled/QAT P or T0034's own-action scorer. A completed training run or code PR alone is not admission evidence. Do not modify T0032 or start duplicate training. Freeze each candidate's carrier, feature/history schema, model/calibration identity, depth, coefficients and complete-turn budgets before confirmation.

Retain the existing confirmation requirement: at least 10,000 independent map groups over counts 2-15, at least 500 per count, paired fixtures and three independently trained seeds for trained finalists. Keep all faults/timeouts, equal count weighting and the production incumbent. Select the fresh confirmation seed only after every included finalist is frozen; do not reuse consumed tuning or training seeds. Report benefit, harm and not-shown separately; no offline prediction result is a BT-strength claim.

## Done when

Confirmation with at least 10,000 independent map groups over counts 2-15 (at least 500 per count), three independently trained seeds and paired fixtures, reported under `bots/p_only/report/confirm/`.

## Handoff

*2026-10-18 19:39Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). The P-only line is paused (its prerequisites T0001 and T0005 were dropped the same day); the agent_v2 carrier is the active line. Reopen as a new task if P-only work resumes.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-10 19:25Z claude@caspers-macbook: note: Candidates for this confirmation may come from T0013 (P feature family p3, equal-time screen) and T0014 (V pilot at fixed XXL P: retrained v2/v3 M, direct 4XL with the Route A bound, distilled). Their finalists arrive frozen (carrier, sizes, coefficients, budgets), with three independently trained seeds where a trained V or P is involved; docs/FEATURE_FAMILIES.md section 8.
- 2026-10-11 01:32Z claude-ci@caspers-macbook: amended depends -> T0004, T0005, T0014; title -> P-only phase 5: freeze the finalists and confirm them; goal
- 2026-10-12 05:58Z codex-rl-plan@caspers-macbook: amended depends -> T0004, T0005, T0013; goal
- 2026-10-12 05:58Z codex-rl-plan@caspers-macbook: note: Casper authorized retiring T0012/T0014; removed the V-pilot dependency and added the P-feature deployment screen T0013. Future distilled-P or T0034 candidates need separate deployment admission before this confirmation.
- 2026-10-18 19:39Z claude-fervent@cloud: dropped
