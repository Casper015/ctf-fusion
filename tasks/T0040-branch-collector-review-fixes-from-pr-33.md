---
id: T0040
title: "Branch collector review fixes from PR #33: input-bound resume and root-turn fault diagnostics (code only)"
status: open
owner: ""
needs: []
writes: [training/decision_branch_collect.py, tests/test_decision_branch_collect.py]
depends: [T0036]
land: pr
branch: ""
updated: 2026-10-12
---

## Goal

Fix the two findings of Casper's review of PR #33 (T0036), which merged with changes still required. Code only: no formal collection or host timing. [P1] training/decision_branch_collect.py resumes a row when collector_identity matches and is_complete sees every (candidate, repeat); nothing binds the row to its input. Reproduced: collect once, change the selected root's logged incumbent refined score 2.5 -> 123.5 with match_id/turn/binaries/config unchanged, collect again -> complete_before=1, played=0, old labels kept. Store in each row (completed and dropped) a digest of the exact input match bytes/content, the root block, the selected complete candidate orders and their aligned carrier scores, and compare them before resuming; a mismatch moves the row aside and replays it (or refuses), never silently reuses it. [P2] At t == turn every bot is still queried under the official limit, but its reply is replaced by the recorded/forced order; faults are serialized only for t > turn and prefix_faults counts only t < turn, so a root-turn timeout or malformed computed reply disappears. Keep the forced root orders and the candidate set unchanged, and add retained discarded_root_computed_faults (or all-phase) counters. Regression tests in tests/test_decision_branch_collect.py: changed score, changed order, changed root block and changed prefix each force replay; a root-turn timeout and a malformed root reply are recorded while the engine root action stays forced.

## Done when

PR merged with both fixes and the regression tests; Casper's review of PR #33 answered; T0037 may then start.

## Handoff

Not started.

## Log

- 2026-10-12 16:06Z claude@caspers-macbook: created
