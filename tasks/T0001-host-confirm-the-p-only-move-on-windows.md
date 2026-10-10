---
id: T0001
title: "Host: confirm the p_only move on Windows"
status: open
owner: ""
needs: [host]
writes: [docs/PATH_MAP.md]
depends: []
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

`p_only/` moved to `bots/p_only/` in c86cdc4. On the Mac all 27 bundled variants stayed byte-identical and every test passed, but nothing has been compiled or run on the host yet. Confirm on the host that the move changed no bot.

Spec: [docs/PATH_MAP.md](../docs/PATH_MAP.md#the-p_only-move-and-bot-identities).

## Done when

- `python -m unittest discover -s bots/p_only/tests -v` passes on Windows, including the engine comparison.
- `python bots/p_only/tools/build.py --stats` gives the same `bundle_sha256` and `binary_sha256` per variant as `bots/p_only/report/screen/manifest.json` (same compiler).
- One line in docs/PATH_MAP.md records the result and the commit checked.

## Handoff

Not started. Do this before T0002: if anything differs, the screens must not start.

Any screen started before c86cdc4 cannot resume (the runner identity changed); restart it from scratch.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
