---
id: T0083
title: "Host: release gate, general PPO policy G versus current default D"
status: active
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [reports/host/agent_v2_release_gate/, profiles/agent_v2_release_gate.json, docs/AGENT_V2_RELEASE_GATE_SEEDS.md]
depends: [T0077]
land: direct
branch: antigravity/t0083-host-release-gate-general-ppo-policy-g
updated: 2026-10-17
---

## Goal

Release gate before agent_v2's production default changes. T0077 showed G = agent_v2_rl_general_csp beats S and I, and T0079 showed search loses to direct play, but G never played the current default D = agent_v2_p2xxl_direct on the same fixtures: they come from different screens and differ in actor (the init actor correlates ~0.995 with aligned_p2_large, not identical) and P bank (CSP vs XXL). Arms: G, D, plus an A/A copy of G; both actor-direct, no search. 96 fixtures per count 2-15 = 1,344 paired fixtures, 4,032 formal games. Proposed seeds: 20262140 preflight, 20262141 screen (free on 2026-10-17; 20262145/46 are taken). Primary contrast G-D at 95%, single comparison; per-count rows descriptive. Use the exact binaries and bundle hashes that PR 69 / T0082 would release, so the gate binds the deployed artifact. Promotion rule, frozen before play: G becomes the default only if G-D 'helps' (95% interval above 0); 'not shown' or 'hurts' keeps D. Shared evaluation contract (N2, N3, N5): fresh paired fixtures share map, flag randomness, seat and opponent schedule across arms; counts 2-15 weighted equally; held-out roster Fable/MiMo/Kimi/GLM/Player4 (profiles/meow_rl_screen.json); exactly eight workers under task.py lock cpu, no GPU training running, 2,000 ms first / 50 ms later replies. Preflight every arm on 16 fixtures at each count 2/5/8/15 (zero all-seat faults, complete telemetry, first reply < 1,800 ms, steady reply < 48 ms); a failure is recorded and blocks the screen, never fewer workers. Freeze binary/model/engine/runner hashes, profile and schedule before play; finish the fixed budget without outcome-based stopping; stop only on integrity or safety failure. Map-cluster 95% intervals, margin zero, multiplicity via the runner's multiplicity field (bots/p_only/tools/run_screen.py). 2-team fixtures were saturated vs this roster in T0057/T0073 (zero difference, zero spread): keep the 14-count primary, and also report a predeclared descriptive equal-weight 3-15 estimate and the 2-team band separately. Seed bases below were free on 2026-10-17: recheck every ledger and derived-seed collision, then reserve before play.

## Done when

Seeds reserved before play; passed preflight (or blocked handoff); complete receipts under reports/host/agent_v2_release_gate/ with the G-D verdict and interval, A/A agreement, the 3-15 and 2-team descriptive rows, fault and timing totals and the pinned release hashes; the verdict and the frozen promotion rule recorded in the T0082 log, and profiles/agent_v2.json changed only through T0082 if G-D helps.

## Handoff

Not started.

## Log

- 2026-10-17 08:58Z claude-review@caspers-macbook: created
- 2026-10-17 09:08Z antigravity@desktop-g60ojud: claimed
- 2026-10-17 09:13Z antigravity@desktop-g60ojud: note: preflight passed at seed 20262140 (128 games, 0 faults/late, max steady reply < 20 ms). Launching formal 4,032-game paired release gate screen under CPU lock (seed 20262141, 8 workers).
