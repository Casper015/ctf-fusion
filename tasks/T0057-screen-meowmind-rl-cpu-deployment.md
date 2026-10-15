---
id: T0057
title: "Screen Meowmind RL CPU deployment"
status: open
owner: ""
needs: [host, ai-test]
writes: [reports/host/meow_rl_screen/, docs/MEOWMIND_RL_SCREEN_SEEDS.md]
depends: [T0056]
land: direct
branch: ""
updated: 2026-10-15
---

## Goal

Measure the frozen actor-direct final large-V PPO candidate against its identical warm-start control, with A/A, following docs/MEOWMIND_RL_PLAN.md. Preflight full CPU reply limits and then execute exactly 392 fresh paired fixtures (28 per count 2-15), three arms, 1176 games. Search, original Meowmind, production and small-V comparisons require a separately frozen secondary schedule.

## Done when

Use CPU lease with GPU idle. Fresh seed receipts, immutable binary/config hashes, interleaved schedule, all arms and all-seat fault accounting, 2s/50ms timing and raw engine outcomes audited. Equal-count paired map-cluster CI, A/A check and backup receipts published. Retain negative results and do not extend after looking at outcomes. No automatic default promotion; positive screened effect still requires replicated training/evaluation.

## Handoff

Not started.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
