---
id: T0042
title: "Decision data diagnostics: root-matched noise correction, repeat IDs, per-phase faults"
status: open
owner: ""
needs: []
writes: [training/decision_dataset.py, training/decision_noise.py, training/evaluate_decisions.py, tests/test_decision_learning.py, tests/test_decision_diagnostics.py, docs/RL_PLAN.md]
depends: []
land: pr
branch: ""
updated: 2026-10-14
---

## Goal

Stage 1 of Casper's own-action scorer preparation (2026-10-14). Compute noise corrections over exactly the roots and candidates of the between-action statistic; report raw between-action, within-action and paired candidate-minus-incumbent repeat variance separately with their covariance and independence assumptions; keep original repeat IDs (slot = repeat id) so paired comparisons and even/odd headroom partitions are unchanged by missing outcomes; keep prefix, discarded root-turn and continuation faults apart (cand_faults is continuation only); keep failed outcomes, unsupported candidates, missing labels and dropped roots inspectable; preserve source-bound receipts, one-carrier identity, map splits and the frozen test gate. Regression tests include T0038's counterexample. Code and synthetic tests only; no collection, training or T0037 frozen file is touched.

## Done when

PR merged with the fixes, documentation in docs/RL_PLAN.md section 3, and regression tests (including the unsupported-root counterexample) passing.

## Handoff

Not started.

## Log

- 2026-10-14 19:41Z claude@caspers-macbook: created
