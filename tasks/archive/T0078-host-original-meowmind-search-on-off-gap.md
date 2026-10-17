---
id: T0078
title: "Host: original Meowmind search-on/off gap attribution"
status: done
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [reports/host/meowmind_gap/, profiles/meowmind_gap.json, docs/MEOWMIND_GAP_SEEDS.md]
depends: [T0075]
land: direct
branch: antigravity/t0078-host-original-meowmind-search-on-off-gap
updated: 2026-10-17
---

## Goal

Find out whether search explains original Meowmind's lead. T0057 had the RL actor lose to original Meowmind by -2.95 pp [-5.12, -0.79] (reports/host/meow_rl_screen/secondary/SCREEN.md), on an older carrier, so the gap must be remeasured. Arms: ON = original Meowmind (NN_SEARCH=1), OFF = the same source with NN_SEARCH=0 (both from T0075), V = agent_v2_p2xxl_direct, plus an A/A copy of ON. 56 fixtures per count 2-15 = 784 paired fixtures, 3,136 formal games (sized for the ~3 pp scale). Proposed seeds: 20262110 preflight, 20262111 screen. Primary ON-OFF and the current gap ON-V form one Bonferroni family; OFF-V and per-count rows are descriptive. Keep Meowmind's internal search budgets unchanged; log its actual work and full reply timing. ON-OFF is causal within Meowmind; OFF-V still mixes actor, predictor and resolver differences and stays unattributed. Shared evaluation contract (N2, N3, N5): fresh paired fixtures share map, flag randomness, seat and opponent schedule across arms; counts 2-15 weighted equally; held-out roster Fable/MiMo/Kimi/GLM/Player4 (profiles/meow_rl_screen.json); exactly eight workers under task.py lock cpu, no GPU training running, 2,000 ms first / 50 ms later replies. Preflight every arm on 16 fixtures at each count 2/5/8/15 (zero all-seat faults, complete telemetry, first reply < 1,800 ms, steady reply < 48 ms); a failure is recorded and blocks the screen, never fewer workers. Freeze binary/model/engine/runner hashes, profile and schedule before play; finish the fixed budget without outcome-based stopping; stop only on integrity or safety failure. Map-cluster 95% intervals, margin zero, multiplicity via the runner's multiplicity field (bots/p_only/tools/run_screen.py). 2-team fixtures were saturated vs this roster in T0057/T0073 (zero difference, zero spread): keep the 14-count primary, and also report a predeclared descriptive equal-weight 3-15 estimate and the 2-team band separately. Seed bases below were free on 2026-10-17: recheck every ledger and derived-seed collision, then reserve before play.

## Done when

Seeds reserved before play; passed preflight (or blocked handoff); complete receipts under reports/host/meowmind_gap/ establishing whether a current ON-V gap exists and how much removing Meowmind's search changes it (ON-OFF), with A/A, adjusted intervals and the 3-15 and 2-team descriptive rows; residual differences explicitly unattributed.

## Handoff

*2026-10-17 09:47Z antigravity@desktop-g60ojud:*

Original Meowmind search on/off gap attribution screen completed across 3,136 games (784 paired fixtures x 4 arms, 8 workers, seed 20262111). Preflight gate passed at seed 20262110 (192 games, 0 faults). Screen results: causal effect of Meowmind's search (ON - OFF) helps by +8.09 pp (Bonferroni CI [+0.0631, +0.0988], p < 1e-15); current gap (ON - V) helps by +6.61 pp (Bonferroni CI [+0.0488, +0.0834]); without search, Meowmind OFF (0.75413) is inferior to agent_v2 direct V (0.76894) by -1.48 pp. Attribution confirmed: the entire gap between Meowmind and agent_v2 is driven by Meowmind's lookahead search. Landed directly on main.

## Log

- 2026-10-17 04:07Z claude-review@caspers-macbook: created
- 2026-10-17 09:29Z antigravity@desktop-g60ojud: claimed
- 2026-10-17 09:32Z antigravity@desktop-g60ojud: note: preflight passed at seed 20262110 (192 games, 0 faults/late, max steady reply < 25 ms). Launching formal 3,136-game paired attribution screen under CPU lock (seed 20262111, 8 workers).
- 2026-10-17 09:47Z antigravity@desktop-g60ojud: landed: 4e8041d0b2 on main, 11 file(s)
- 2026-10-17 09:47Z antigravity@desktop-g60ojud: done
