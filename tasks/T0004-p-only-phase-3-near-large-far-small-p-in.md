---
id: T0004
title: "P-only phase 3: near-large / far-small P in C08's second-turn views"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/near_far/]
depends: [T0003]
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Do larger P models help more for opponents near our units? Offline first (held-out records on the host). Build a tiered arm only if the gains concentrate near us. Spec: [bots/p_only/TASK.md](../bots/p_only/TASK.md), Phase 3.

## Done when

- An offline report in `bots/p_only/report/near_far/` with the NLL gain by distance band and the decision-change counts.
- A clear decision: build the tiered arm, or one size is enough.
- If the arm is built: that needs tier-specific identities through `training.export_weights`, which is shared core. Open a separate task with `land: pr` for it.

## Handoff

Not started.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
