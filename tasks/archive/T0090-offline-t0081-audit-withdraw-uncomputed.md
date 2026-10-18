---
id: T0090
title: "Offline T0081 audit: withdraw uncomputed diagnostics and regenerate from receipts"
status: done
owner: claude-review@caspers-macbook
needs: []
writes: [tools/screens/p3_search_host/, reports/host/agent_v2_p3_search/diagnostics/, reports/host/agent_v2_p3_search/CORRECTION.md, reports/host/agent_v2_p3_search/REPORT.md]
depends: [T0081]
land: pr
branch: claude-review/t0090-offline-t0081-audit-withdraw-uncomputed
updated: 2026-10-18
---

## Goal

@/private/tmp/claude-501/-Users-casper-code-ctf-fusion--claude-worktrees-codex-access-check-101f27/f01f384a-dadd-4ab8-bec7-2b0ecfbb77ec/scratchpad/t90_goal.md

## Done when

@/private/tmp/claude-501/-Users-casper-code-ctf-fusion--claude-worktrees-codex-access-check-101f27/f01f384a-dadd-4ab8-bec7-2b0ecfbb77ec/scratchpad/t90_done.md

## Handoff

*2026-10-18 05:08Z claude-review@caspers-macbook:*

Merged PR 70 (b251e2b). diagnose.py fails closed and computes only from PREFLIGHT.json; uncomputed T0081 diagnostics withdrawn (reports/host/agent_v2_p3_search/CORRECTION.md). Preflight completion P3/P2 search: n05 0.548/0.972, n15 0.025/0.025; overrides 37-70% of turns at 2-5 teams. Host-only audit remains in T0084.

## Log

- 2026-10-18 03:40Z claude-review@caspers-macbook: created
- 2026-10-18 03:41Z claude-review@caspers-macbook: claimed
- 2026-10-18 03:59Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/70 (head 491c1ef). Local tests 8/8, core suite 474 OK (60 skipped). Ready for review.
- 2026-10-18 04:29Z claude-review@caspers-macbook: note: Codex (gpt-6.1-sol high) review of 491c1ef posted on PR 70 as changes requested: 3 must-fix (counter validation, coverage checks, T0079 figures in section 5) and 6 should-fix, all confirmed and fixed in 39809e2. 10 local tests pass.
- 2026-10-18 04:47Z claude-review@caspers-macbook: note: Codex re-review of 39809e2 posted on PR 70 (changes requested): refused-turn accounting, zero-turn games, cross-count fixture ids, critic/int checks. All confirmed and fixed in 4ca2287.
- 2026-10-18 05:08Z claude-review@caspers-macbook: done
