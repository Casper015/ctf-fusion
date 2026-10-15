---
id: T0057
title: "Screen Meowmind RL CPU deployment"
status: done
owner: codex-decision-pilot@desktop-g60ojud
needs: [host, ai-test]
writes: [bots/p_only/tools/preflight_meow_rl.py, docs/MEOWMIND_RL_SCREEN_SEEDS.md, profiles/meow_rl_screen_primary.json, profiles/meow_rl_screen_secondary.json, reports/host/meow_rl_screen/]
depends: [T0056]
land: direct
branch: codex-decision-pilot/t0057-screen-meowmind-rl-cpu-deployment
updated: 2026-10-15
---

## Goal

Measure the frozen actor-direct final large-V PPO candidate against its identical warm-start control, with A/A, following docs/MEOWMIND_RL_PLAN.md. Preflight full CPU reply limits and then execute exactly 392 fresh paired fixtures (28 per count 2-15), three arms, 1176 games. Search, original Meowmind, production and small-V comparisons require a separately frozen secondary schedule.

## Done when

Use CPU lease with GPU idle. Fresh seed receipts, immutable binary/config hashes, interleaved schedule, all arms and all-seat fault accounting, 2s/50ms timing and raw engine outcomes audited. Equal-count paired map-cluster CI, A/A check and backup receipts published. Retain negative results and do not extend after looking at outcomes. No automatic default promotion; positive screened effect still requires replicated training/evaluation.

## Handoff

*2026-10-15 12:41Z codex-decision-pilot@desktop-g60ojud:*

Preflight on desktop-g60ojud admitted 6 workers, 30 ms search deadline (15-team max latency 13.99 ms). Screen results against held-out reference roster (Fable, MiMo, Kimi, GLM, Player4) with 0 faults across all 4,312 games: Primary screen (+0.0318 [+0.0080, +0.0557] helps vs warm start), Secondary screen (large V helps vs small V +0.0325; search hurts -0.1479; P3 distill vs XXL -0.0045 not shown; Meowmind +0.0295 and P2 +0.0304 stronger than actor-direct). 0 noise across both A/A checks (392/392 identical).

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
- 2026-10-15 12:06Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 12:35Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/p_only/tools/preflight_meow_rl.py, docs/MEOWMIND_RL_SCREEN_SEEDS.md, profiles/meow_rl_screen_primary.json, profiles/meow_rl_screen_secondary.json, reports/host/meow_rl_screen/
- 2026-10-15 12:40Z codex-decision-pilot@desktop-g60ojud: landed: 36b84897a0 on main, 17 file(s)
- 2026-10-15 12:41Z codex-decision-pilot@desktop-g60ojud: done
