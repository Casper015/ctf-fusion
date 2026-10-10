---
id: T0020
title: "Fix C08 per-branch search deadlines and branch-regret comparison validity"
status: open
owner: ""
needs: []
writes: [bots/p_only/tools/, bots/p_only/bots/, bots/p_only/profiles/, bots/p_only/tests/, docs/SEARCH_PORTFOLIO.md]
depends: []
land: pr
branch: ""
updated: 2026-10-10
---

## Goal

Correctness fixes to T0018 (PR #10, reviewed at cf16c05) before T0019 runs. (1) search::rank in the C08 family carrier (generator bots/p_only/tools/derive/search_experiment.py): in PONLY_PLY2_MODE 1 and 2 every plan started a ply-2 prediction after a single per-sweep deadline check; check the deadline before each prediction and after the expensive work, never commit a partial sweep (no unequal committed sample counts, main or diagnostic), keep counters truthful, return the last complete comparison or the existing fallback, uniform and halving paths both; regression tests with a controlled clock/prediction source. (2) branch_portfolio.py / branch_regret.py: a malformed or missing comparison reply or branch must never count as 'same as incumbent' with zero difference; validate and canonicalize the reply, record its status (identical, different, malformed, timeout, missing branch), report exclusions and the valid-root denominator; tests for every case. Also move screen_search.json off seed 20261220 (consumed by T0016) to an unseen seed. Models, features, depth, coefficients and production defaults unchanged; no edits under training/ or tests/.

## Done when

PR merged with the generator fix, regenerated carriers, branch tool fixes, regression tests, a fresh screen seed and docs; both suites pass; T0019's Handoff/log names the corrected commands.

## Handoff

Not started.

## Log

- 2026-10-10 23:45Z claude@caspers-macbook: created
