---
id: T0002
title: "P-only stage 1: capacity and sampling screen at two plies"
status: dropped
owner: ""
needs: [host]
writes: [bots/p_only/report/capacity/]
depends: [T0001]
land: direct
branch: ""
updated: 2026-10-18
---

## Goal

Does a larger P, or more search samples, make the complete bot stronger under the 50 ms deadline? Stage 1 of [bots/p_only/EXPERIMENTS.md](../../bots/p_only/EXPERIMENTS.md): the 2-ply control, C08 XL and XXL at fixed time and at 256 fixed samples, and v12 XL and XXL, with A/A copies of the finalists. 2,520 games, seed 20261107.

Rules: [bots/p_only/TASK.md](../../bots/p_only/TASK.md) (Rules, Phase 2b).

## Done when

- `bots/p_only/report/capacity/` holds SCREEN.md and its JSON, committed.
- Every comparison has its interval and verdict (helps, hurts or not shown), plus the truncated share for fixed-sample arms.
- A summary in this task's Handoff: what was measured, the verdicts, the timing, what is still open.

## Handoff

*2026-10-18 19:39Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). The P-only line is paused (its prerequisites T0001 and T0005 were dropped the same day); the agent_v2 carrier is the active line. Reopen as a new task if P-only work resumes.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-18 19:39Z claude-fervent@cloud: dropped
