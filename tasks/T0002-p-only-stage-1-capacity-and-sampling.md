---
id: T0002
title: "P-only stage 1: capacity and sampling screen at two plies"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/capacity/]
depends: [T0001]
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Does a larger P, or more search samples, make the complete bot stronger under the 50 ms deadline? Stage 1 of [bots/p_only/EXPERIMENTS.md](../bots/p_only/EXPERIMENTS.md): the 2-ply control, C08 XL and XXL at fixed time and at 256 fixed samples, and v12 XL and XXL, with A/A copies of the finalists. 2,520 games, seed 20261107.

Rules: [bots/p_only/TASK.md](../bots/p_only/TASK.md) (Rules, Phase 2b).

## Done when

- `bots/p_only/report/capacity/` holds SCREEN.md and its JSON, committed.
- Every comparison has its interval and verdict (helps, hurts or not shown), plus the truncated share for fixed-sample arms.
- A summary in this task's Handoff: what was measured, the verdicts, the timing, what is still open.

## Handoff

Not started. Commands (host):

```
python tools/repo/task.py lock cpu --task T0002 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_capacity.json --dry-run
python tools/repo/task.py lock cpu --task T0002 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_capacity.json
```

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
