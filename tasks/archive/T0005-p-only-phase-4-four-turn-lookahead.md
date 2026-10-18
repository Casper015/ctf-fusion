---
id: T0005
title: "P-only phase 4: four-turn lookahead"
status: dropped
owner: ""
needs: [host]
writes: [bots/p_only/]
depends: [T0003, T0019]
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

Extend C08 P-only from two to four simulated turns with the same single P, each state predicted from its own simulated parent. Spec: [bots/p_only/TASK.md](../../bots/p_only/TASK.md), Phase 4.

Phase 4 step 1 (compare the one second-turn prediction shared by every candidate plan with a prediction per candidate: cost per turn, how often C08's chosen move changes) is the same work as T0018 (PONLY_PLY2_MODE: per-branch ply-2 predictions with a diagnostic mode, merged in PR #10, fixed in PR #13) measured by T0019 step 1 (fixed-work diagnostics). Start from T0019's report instead of repeating it; this task does steps 2 to 6.

## Done when

- Depth is a setting from 2 to 4. The contract test proves that branch order and count never change the real root, and keeps the engine comparison.
- Fixed-sample and fixed-time screens against the 4-turn scripted control and the 2-turn P-only arm, committed under `bots/p_only/report/`.
- The bot changes land through a pull request with that evidence.

## Handoff

*2026-10-18 19:37Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). The P-only line is paused; the agent_v2 carrier is the active line. Reopen as a new task if P-only work resumes.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-11 01:32Z claude-ci@caspers-macbook: amended depends -> T0003, T0019; goal
- 2026-10-18 19:37Z claude-fervent@cloud: dropped
