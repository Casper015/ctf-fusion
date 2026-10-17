---
id: T0079
title: "Host: gated wider agent_v2 search versus direct and ungated wider search"
status: active
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [reports/host/agent_v2_search_gate/, profiles/agent_v2_search_gate.json, docs/AGENT_V2_SEARCH_GATE_SEEDS.md]
depends: [T0076]
land: direct
branch: antigravity/t0079-host-gated-wider-agent-v2-search-versus
updated: 2026-10-17
---

## Goal

Screen T0076's frozen search repair. Arms: D = agent_v2_p2xxl_direct, GW = the frozen gated-wide search, UW = the otherwise identical ungated-wide search, plus an A/A copy of GW. 96 fixtures per count 2-15 = 1,344 paired fixtures, 5,376 formal games. Proposed seeds: 20262130 preflight, 20262131 screen. Primary GW-D and supporting GW-UW share one Bonferroni family; per-count rows are descriptive. tau and breadth come only from T0076; no adjustment from this screen's data. Record completed work, overrides, fallbacks and deadline hits per arm. Cancel play if T0076 failed its diagnostic gate. If GW-D helps, a separate fresh screen of CSP vs XXL P inside the gated search follows (T0073 only tested P for attack targeting in direct play); otherwise that test is cancelled. Shared evaluation contract (N2, N3, N5): fresh paired fixtures share map, flag randomness, seat and opponent schedule across arms; counts 2-15 weighted equally; held-out roster Fable/MiMo/Kimi/GLM/Player4 (profiles/meow_rl_screen.json); exactly eight workers under task.py lock cpu, no GPU training running, 2,000 ms first / 50 ms later replies. Preflight every arm on 16 fixtures at each count 2/5/8/15 (zero all-seat faults, complete telemetry, first reply < 1,800 ms, steady reply < 48 ms); a failure is recorded and blocks the screen, never fewer workers. Freeze binary/model/engine/runner hashes, profile and schedule before play; finish the fixed budget without outcome-based stopping; stop only on integrity or safety failure. Map-cluster 95% intervals, margin zero, multiplicity via the runner's multiplicity field (bots/p_only/tools/run_screen.py). 2-team fixtures were saturated vs this roster in T0057/T0073 (zero difference, zero spread): keep the 14-count primary, and also report a predeclared descriptive equal-weight 3-15 estimate and the 2-team band separately. Seed bases below were free on 2026-10-17: recheck every ledger and derived-seed collision, then reserve before play.

## Done when

Seeds reserved before play; passed preflight (or blocked handoff); complete receipts under reports/host/agent_v2_search_gate/ with helps/hurts/not-shown verdicts on adjusted map-cluster intervals, A/A, override and fallback counts, and the 3-15 and 2-team descriptive rows, or the documented blocking failure.

## Handoff

Not started.

## Log

- 2026-10-17 04:07Z claude-review@caspers-macbook: created
- 2026-10-17 05:28Z claude-t76@caspers-macbook: note: from T0076 (PR 67): GW = agent_v2_p2xxl_gw8x4_fp32 (tau 0.002), UW = agent_v2_p2xxl_uw8x4_fp32, D included in bots/agent_v2/search_gate/variants.json. Admission is unchanged: every trajectory must be admitted before 16.5 ms of the 30 ms deadline, and the worst-seen cost never decays, so one slow trajectory turns search off for the rest of the game. A Mac smoke run (not evidence) completed few 8x4 trees and almost none at 15 teams. Preflight must count search_used per arm and count, or GW/UW may play D's order.
- 2026-10-17 06:58Z antigravity@desktop-g60ojud: claimed
