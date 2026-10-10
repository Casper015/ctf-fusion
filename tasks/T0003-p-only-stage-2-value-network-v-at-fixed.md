---
id: T0003
title: "P-only stage 2: value network V at fixed P"
status: done
owner: codex@desktop-g60ojud
needs: [host]
writes: [bots/p_only/report/value/]
depends: [T0002]
land: direct
branch: codex/t0003-p-only-stage-2-value-network-v-at-fixed
updated: 2026-10-10
---

## Goal

Does V help when P is held fixed? Stage 2 of [bots/p_only/EXPERIMENTS.md](../bots/p_only/EXPERIMENTS.md): P-XL and P-XXL, each with the same V-M off and on (mixed model sets `inference/models_pxl_vm/`, `models_pxxl_vm/`). 2,240 games, seed 20261108.

## Done when

- `bots/p_only/report/value/` committed.
- Each comparison reported with its interval, and V's call counts per path.
- The summary in Handoff.

## Handoff

*2026-10-10 10:02Z codex@desktop-g60ojud:*

result in bots/p_only/report/value/SCREEN.md and STAGE2_REPORT.md; V benefit not shown overall (+0.010 C08 XL, +0.018 C08 XXL, +0.012 v12 XL, -0.002 v12 XXL); helps in C08 XL 11-15 (+0.037); lack of gain diagnosed as continuation policy shift and coarse features rather than starvation or capacity

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-10 09:23Z codex@desktop-g60ojud: claimed (forced)
- 2026-10-10 09:44Z codex@desktop-g60ojud: note: completed screen_value: 2,240 games, 8 arms, seed 20261108; result in bots/p_only/report/value/SCREEN.md
- 2026-10-10 10:02Z codex@desktop-g60ojud: landed: bdcd218a7c on main, 7 file(s)
- 2026-10-10 10:02Z codex@desktop-g60ojud: done
