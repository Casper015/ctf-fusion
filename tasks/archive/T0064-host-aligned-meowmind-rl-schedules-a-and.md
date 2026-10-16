---
id: T0064
title: "Host: aligned Meowmind RL schedules A and B (3,920 games), verdicts and conclusion (T0061 runbook 9-10)"
status: done
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

*2026-10-16 00:37Z codex-decision-pilot@desktop-g60ojud:*

Aligned schedules A and B fully executed (3920 games, 8 workers, 0 faults/timeouts). Schedule A confirmed direct policy RL gain over warm start: +3.84 pp [+1.66, +6.01] (helps). Schedule B confirmed WideAct16 eliminated quantization loss (+10.10 pp over r16, matching FP32 within 0.7 pp); both FP32 (-3.91 pp) and WideAct16 (-4.66 pp) show residual search degradation against direct play. VERDICTS and CONCLUSION.md committed. Off-machine backups verified: D:/backups/ctf_fusion/meow_rl_aligned_screens_20261015.tar.gz (200993626 bytes, SHA-256: 07dd41311768f4a76b981bd9ead4e3e418734f4ce7eb9f51fa833865e12e89c7).

## Log

- 2026-10-15 22:19Z claude-review@caspers-macbook: created
- 2026-10-15 22:20Z claude-review@caspers-macbook: amended goal
- 2026-10-16 00:00Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-16 00:36Z codex-decision-pilot@desktop-g60ojud: landed: 3b1d7b51bc on main, 15 file(s)
- 2026-10-16 00:37Z codex-decision-pilot@desktop-g60ojud: done
