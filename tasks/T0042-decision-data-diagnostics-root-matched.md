---
id: T0042
title: "Decision data diagnostics: root-matched noise correction, repeat IDs, per-phase faults"
status: review
owner: claude@caspers-macbook
needs: []
writes: [training/decision_dataset.py, training/decision_noise.py, training/evaluate_decisions.py, tests/test_decision_learning.py, tests/test_decision_diagnostics.py, docs/RL_PLAN.md]
depends: []
land: pr
branch: claude/t0042-decision-data-diagnostics-root-matched
updated: 2026-10-14
---

## Goal

Stage 1 of Casper's own-action scorer preparation (2026-10-14). Compute noise corrections over exactly the roots and candidates of the between-action statistic; report raw between-action, within-action and paired candidate-minus-incumbent repeat variance separately with their covariance and independence assumptions; keep original repeat IDs (slot = repeat id) so paired comparisons and even/odd headroom partitions are unchanged by missing outcomes; keep prefix, discarded root-turn and continuation faults apart (cand_faults is continuation only); keep failed outcomes, unsupported candidates, missing labels and dropped roots inspectable; preserve source-bound receipts, one-carrier identity, map splits and the frozen test gate. Regression tests include T0038's counterexample. Code and synthetic tests only; no collection, training or T0037 frozen file is touched.

## Done when

PR merged with the fixes, documentation in docs/RL_PLAN.md section 3, and regression tests (including the unsupported-root counterexample) passing.

## Handoff

*2026-10-14 19:47Z claude@caspers-macbook:*

PR #39 (branch claude/t0042-decision-data-diagnostics-root-matched, commit 05d3371). Done: noise statistics over one noise-root population (T0038 counterexample is a regression test), paired gap variance and implied covariance with stated assumptions, dataset schema ctf-fusion-decision-2 keeping repeat ids, failed repeats and per-phase faults, headroom by repeat id, RL_PLAN section 3. Verified: core suite 311 OK. Next: review/merge; T0038 builds its dataset with this schema (python -m training.decision_dataset ... then python -m training.decision_noise --data <out> --out <report>). Not checked: real noise levels (synthetic outcomes only).

## Log

- 2026-10-14 19:41Z claude@caspers-macbook: created
- 2026-10-14 19:41Z claude@caspers-macbook: claimed
- 2026-10-14 19:47Z claude@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/39 opened.
- 2026-10-14 19:47Z claude@caspers-macbook: handoff -> review
