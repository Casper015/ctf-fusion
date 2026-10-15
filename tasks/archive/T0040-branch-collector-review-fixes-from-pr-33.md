---
id: T0040
title: "Branch collector review fixes from PR #33: input-bound resume and root-turn fault diagnostics (code only)"
status: done
owner: codex-t0037-report@caspers-macbook
needs: []
writes: [training/decision_branch_collect.py, tests/test_decision_branch_collect.py]
depends: [T0036]
land: pr
branch: claude/t0040-branch-collector-review-fixes-from-pr-33
updated: 2026-10-12
---

## Goal

Fix the two findings of Casper's review of PR #33 (T0036), which merged with changes still required. Code only: no formal collection or host timing. [P1] training/decision_branch_collect.py resumes a row when collector_identity matches and is_complete sees every (candidate, repeat); nothing binds the row to its input. Reproduced: collect once, change the selected root's logged incumbent refined score 2.5 -> 123.5 with match_id/turn/binaries/config unchanged, collect again -> complete_before=1, played=0, old labels kept. Store in each row (completed and dropped) a digest of the exact input match bytes/content, the root block, the selected complete candidate orders and their aligned carrier scores, and compare them before resuming; a mismatch moves the row aside and replays it (or refuses), never silently reuses it. [P2] At t == turn every bot is still queried under the official limit, but its reply is replaced by the recorded/forced order; faults are serialized only for t > turn and prefix_faults counts only t < turn, so a root-turn timeout or malformed computed reply disappears. Keep the forced root orders and the candidate set unchanged, and add retained discarded_root_computed_faults (or all-phase) counters. Regression tests in tests/test_decision_branch_collect.py: changed score, changed order, changed root block and changed prefix each force replay; a root-turn timeout and a malformed root reply are recorded while the engine root action stays forced.

## Done when

PR merged with both fixes and the regression tests; Casper's review of PR #33 answered; T0037 may then start.

## Handoff

*2026-10-12 18:18Z codex-t0037-report@caspers-macbook:*

PR #36 reviewed and merged under Casper explicit request to finish remaining collection. Exact head 9c118942f9c5a249b24605f052751da2f72508ae: input-bound resume/root-fault implementation inspected; 19 focused tests pass; exact-head CI succeeds. Code-only task complete.

## Log

- 2026-10-12 16:06Z claude@caspers-macbook: created
- 2026-10-12 16:09Z codex-review@caspers-macbook: note: Independent review reproduction confirmed: selected incumbent refined score 2.5 -> 123.5 with unchanged match_id/turn/config causes complete_before=1, played=0. Review https://github.com/Casper015/ctf-fusion/pull/33#pullrequestreview-5493229881 also identifies unretained computed root faults. Existing tests pass but do not cover these cases; add regressions before branch-data acceptance. No code edits or host collection performed in this review.
- 2026-10-12 16:15Z claude@caspers-macbook: claimed
- 2026-10-12 16:21Z claude@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/36 (commit on claude/t0040-branch-collector-review-fixes-from-pr-33): input_binding on every row (match, root block, prefix, orders, scores), stale rows set aside and replayed; root-turn computed faults retained. 19 tests OK, the 8 new ones fail on main's collector; core OK.
- 2026-10-12 16:23Z claude@caspers-macbook: handoff -> review
- 2026-10-12 18:18Z codex-t0037-report@caspers-macbook: claimed (forced)
- 2026-10-12 18:18Z codex-t0037-report@caspers-macbook: done