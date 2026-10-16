---
id: T0064
title: "Host: aligned Meowmind RL schedules A and B (3,920 games), verdicts and conclusion (T0061 runbook 9-10)"
status: active
owner: codex-decision-pilot@desktop-g60ojud
needs: [host, ai-test]
writes: [reports/host/meow_rl_aligned/screen_a/, reports/host/meow_rl_aligned/screen_b/, reports/host/meow_rl_aligned/VERDICTS.md, reports/host/meow_rl_aligned/VERDICTS.json, reports/host/meow_rl_aligned/CONCLUSION.md]
depends: [T0063]
land: direct
branch: codex-decision-pilot/t0064-host-aligned-meowmind-rl-schedules-a-and
updated: 2026-10-16
---

## Goal

Run steps 9-10 of the T0061 host runbook on desktop-g60ojud, after T0063's preflight passed and its freeze committed.
1. Under "lock cpu":
       python bots/p_only/tools/run_screen.py --config profiles/meow_rl_aligned_screen_a.json
   then the same with ..._b.json. That is 1,960 games each, 3,920 in all, exactly 8 workers, bound to the preflight receipt.
   - No early stopping or extension.
   - Rerun step 3 only to complete missing fixtures, as run_screen.py documents.
   - Report incomplete fixtures as they are.
2. Generate the verdicts: python bots/meow_rl/aligned/report.py (writes reports/host/meow_rl_aligned/VERDICTS.{md,json}).
3. Back up the raw logs (data/meow_rl_aligned/screen_*) off-machine with verified hashes, and record the receipts.
4. Write reports/host/meow_rl_aligned/CONCLUSION.md (concise, English). It states what improved, what failed, what remains unknown, and whether more training is justified. It must combine:
   - Schedule A: primary at 95%, secondary Bonferroni over 2;
   - Schedule B: Bonferroni over 5;
   - the fixed-work diagnostics;
   - the preflight's practical CPU limits.
   Do not conclude:
   - "RL gain confirmed" from T0057;
   - "zero noise" from A/A;
   - a large-V mechanism from one seed;
   - a strongest bot from a sample mean.
   If search still loses, propose (do not run) a separately budgeted real-engine continuation task with a fresh seed ledger.

## Done when

both screens complete or reported incomplete, VERDICTS and CONCLUSION.md committed, raw-log backup receipts in the log

## Handoff

Not started.

## Log

- 2026-10-15 22:19Z claude-review@caspers-macbook: created
- 2026-10-15 22:20Z claude-review@caspers-macbook: amended goal
- 2026-10-16 00:00Z codex-decision-pilot@desktop-g60ojud: claimed
