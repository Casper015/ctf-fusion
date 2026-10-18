---
id: T0001
title: "Host: confirm the p_only move on Windows"
status: dropped
owner: ""
needs: [host]
writes: [docs/PATH_MAP.md]
depends: []
land: direct
branch: ""
updated: 2026-10-18
---

## Goal

`p_only/` moved to `bots/p_only/` in c86cdc4. On the Mac all 27 bundled variants stayed byte-identical and every test passed, but nothing has been compiled or run on the host yet. Confirm on the host that the move changed no bot.

Spec: [docs/PATH_MAP.md](../../docs/PATH_MAP.md#the-p_only-move-and-bot-identities).

## Done when

- `python -m unittest discover -s bots/p_only/tests -v` passes on Windows, including the engine comparison.
- `python bots/p_only/tools/build.py --stats` gives the same `bundle_sha256` and `binary_sha256` per variant as `bots/p_only/report/screen/manifest.json` (same compiler).
- One line in docs/PATH_MAP.md records the result and the commit checked.

## Handoff

*2026-10-18 19:37Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). The P-only line is paused; the agent_v2 carrier is the active line. Reopen as a new task if P-only work resumes.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-11 01:32Z claude-ci@caspers-macbook: note: Board review (T0024): the check as written can no longer pass. Every variant rebuilt on the host after c86cdc4 changed inputs for other reasons (inference/mlp.hpp in T0016, the carriers in T0009 and T0018), and the post-move manifests that match the pre-move ones reuse old entries without a rebuild. The host already ran the P-only suite after the move (T0016: 20 passed). Question for Casper: drop this task and remove it from T0002's dependencies? T0002 waits on it.
- 2026-10-18 19:37Z claude-fervent@cloud: dropped
