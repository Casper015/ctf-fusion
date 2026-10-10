---
id: T0003
title: "P-only stage 2: value network V at fixed P"
status: active
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

Not started. Command (host): `python tools/repo/task.py lock cpu --task T0003 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_value.json`

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-10 09:23Z codex@desktop-g60ojud: claimed (forced)
- 2026-10-10 09:44Z codex@desktop-g60ojud: note: completed screen_value: 2,240 games, 8 arms, seed 20261108; result in bots/p_only/report/value/SCREEN.md
- 2026-10-10 10:02Z codex@desktop-g60ojud: landed: bdcd218a7c on main, 7 file(s)
