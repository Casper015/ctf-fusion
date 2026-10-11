---
id: T0020
title: "Fix C08 per-branch search deadlines and branch-regret comparison validity"
status: review
owner: claude@caspers-macbook
needs: []
writes: [bots/p_only/tools/, bots/p_only/bots/, bots/p_only/profiles/, bots/p_only/tests/, docs/SEARCH_PORTFOLIO.md]
depends: []
land: pr
branch: claude/t0020-fix-c08-per-branch-search-deadlines-and
updated: 2026-10-11
---

## Goal

Correctness fixes to T0018 (PR #10, reviewed at cf16c05) before T0019 runs. (1) search::rank in the C08 family carrier (generator bots/p_only/tools/derive/search_experiment.py): in PONLY_PLY2_MODE 1 and 2 every plan started a ply-2 prediction after a single per-sweep deadline check; check the deadline before each prediction and after the expensive work, never commit a partial sweep (no unequal committed sample counts, main or diagnostic), keep counters truthful, return the last complete comparison or the existing fallback, uniform and halving paths both; regression tests with a controlled clock/prediction source. (2) branch_portfolio.py / branch_regret.py: a malformed or missing comparison reply or branch must never count as 'same as incumbent' with zero difference; validate and canonicalize the reply, record its status (identical, different, malformed, timeout, missing branch), report exclusions and the valid-root denominator; tests for every case. Also move screen_search.json off seed 20261220 (consumed by T0016) to an unseen seed. Models, features, depth, coefficients and production defaults unchanged; no edits under training/ or tests/.

## Done when

PR merged with the generator fix, regenerated carriers, branch tool fixes, regression tests, a fresh screen seed and docs; both suites pass; T0019's Handoff/log names the corrected commands.

## Handoff

*2026-10-11 00:26Z claude@caspers-macbook:*

Status: in review as https://github.com/Casper015/ctf-fusion/pull/13 (branch claude/t0020-fix-c08-per-branch-search-deadlines-and, commit 0f3bec5). Code and development-machine checks only. No timing or strength measured.

Done:
- search::rank, regenerated from bots/p_only/tools/derive/search_experiment.py:
  - a sample is committed only when its sweep over every plan ends;
  - modes 1 and 2 check the applicable deadline before and after every ply-2 prediction;
  - an expired sweep is discarded and counted (aborted_sweeps, discarded_predictions);
  - after the call's deadline, no further work starts;
  - no completed sample returns no ranking, and the planner keeps its incumbent;
  - mode 0 is unchanged; search::clockHook makes this testable.
- branch_portfolio / branch_regret:
  - replies are compared by canonical complete order and classified: identical, different, malformed, timeout, missing_candidate, missing_branch, none;
  - only identical and different roots enter the statistics, and the report lists the exclusions and the valid-root denominator.
- screen_search.json seed: 20261220 -> 20261230 (20261220 was consumed by T0016).
- Tests: rank_deadline_test.cpp (injected clock) and test_branch_regret.py. The defaults-equivalence test now retries replays that a wall-clock deadline cut.
- Docs: docs/SEARCH_PORTFOLIO.md section 8.

Checks (caspers-macbook, AI_TEST_ROOT set):
- python -m unittest discover -s tests -t . -> 152 tests OK (13 skipped: no torch).
- python -m unittest discover -s bots/p_only/tests -> 43 tests OK, no skips.

Next:
1. A human reviews and merges PR #13, then: python tools/repo/task.py done T0020 --text "merged as <sha>".
2. T0019 can then run A/B/C/D at equal full-turn time. Use seed 20261230 and the lock syntax "task.py lock --task T0019 cpu -- <command>" (see T0019's log).

Unverified:
- Host timing.
- How far one already-running prediction can overrun a deadline; mode 0 can overrun by one shared prediction, as before.
- Samples displaced by per-branch prediction at equal time (about 9.5x the predictions at fixed work).

## Log

- 2026-10-10 23:45Z claude@caspers-macbook: created
- 2026-10-10 23:45Z claude@caspers-macbook: claimed
- 2026-10-11 00:26Z claude@caspers-macbook: handoff -> review
