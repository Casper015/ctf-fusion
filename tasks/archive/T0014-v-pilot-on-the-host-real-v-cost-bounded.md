---
id: T0014
title: "V pilot on the host: real V cost, bounded direct 4XL, distilled V, screen at fixed P"
status: dropped
owner: ""
needs: [host]
writes: [bots/p_only/report/v_pilot/, reports/host/value_cost_DESKTOP-G60OJUD.json, reports/host/value_cost_DESKTOP-G60OJUD.md]
depends: [T0012]
land: direct
branch: ""
updated: 2026-10-12
---

## Goal

Deployment half of the V pilot (docs/FEATURE_FAMILIES.md sections 7 and 8) with the frozen XXL P and the exported sets inference/models_pxxl_vpA..E of T0012. On DESKTOP-G60OJUD after T0012 is merged: (1) V cost with real weights at representative concurrency: python tools/repo/task.py lock cpu --task T0014 -- python tools/host/value_cost.py --set inference/models_pxxl_vm --set inference/models_pxxl_vpB --set inference/models_pxxl_vpD --set inference/models_pxxl_vpE --calls 7 --copies 1 6 15 --out reports/host/value_cost_DESKTOP-G60OJUD (the stage 2 screen measured about 6.6 v12 rollout V calls per turn). (2) Check the Route A bound: the 4XL arms set PONLY_V_DEADLINE_MS 30; if (1) shows a 4XL turn cannot fit the internal budgets, lower it or cap calls (PONLY_V_MAX_CALLS) in bots/p_only/bots/variants/v12_ponly_pxxl_vp{C,D}_v.cpp through a small T0009-style change, never by relaxing the 50 ms limit. (3) Screen at fixed P, V off vs existing V vs new small V vs direct 4XL vs distilled, interleaved with A/A: python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_v_pilot.json --dry-run, then under the CPU lock without --dry-run (7 arms + 1 A/A x 280 fixtures = 2,240 games), tuning seed 20261211. Read the VALUE and VBUDGET stderr counters (V calls, seconds, cut calls, fallback decisions) and full-reply p50/p99/max and faults. (4) Coefficient tuning (PONLY_V_ROLLOUT) only on this tuning seed, then freeze carrier, sizes, coefficients and budgets and hand the finalists to T0006 (confirmation: at least 10,000 independent map groups over counts 2-15, 500 per count, three independently trained V seeds, paired fixtures, unseen seed). Report inconclusive intervals as inconclusive.

## Done when

reports/host/value_cost_DESKTOP-G60OJUD.* and bots/p_only/report/v_pilot/ committed; Handoff names the frozen finalists (or none) with their evidence, and notes them in T0006.

## Handoff

*2026-10-12 05:58Z codex-rl-plan@caspers-macbook:*

Retired the new-V cost/coefficient tuning and direct/distilled 4XL deployment screen after T0012 retirement. Keep production V off. User-authorized retirement on 2026-10-11 (America/Los_Angeles). Existing V experiments, code and collected artifacts remain preserved. This task was not executed and is dropped, not marked successfully completed. Decision-learning preparation continues in T0034; new V work requires an explicit new decision/task.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-11 01:32Z claude-ci@caspers-macbook: note: T0006 no longer repeats the V comparison and tuning of TASK.md Phase 5: that is steps 3 and 4 here. Hand the frozen finalists to T0006, which now only confirms them.
- 2026-10-12 05:58Z codex-rl-plan@caspers-macbook: dropped