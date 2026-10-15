---
id: T0056
title: "Export and gate Meowmind RL CPU bots"
status: open
owner: ""
needs: []
writes: [bots/meow_rl/, inference/meow_rl/, tests/test_meow_rl_export.py, profiles/meow_rl_screen.json, docs/MEOWMIND_RL_DEPLOYMENT.md]
depends: [T0055]
land: pr
branch: ""
updated: 2026-10-15
---

## Goal

Export the frozen pilot actor into an actor-direct CPU bot with unchanged resolver and a matched warm-start control. Add a separately switchable bounded two-ply large-V search arm using consistent utility reward units, explicit cache identities and deadline fallback. Follow docs/MEOWMIND_RL_PLAN.md. No default replacement.

## Done when

Python/native output parity, model/feature/reward-unit identity checks, protocol and fallback tests pass; frozen primary actor-direct vs warm-start plus A/A screen specification prepared with fresh seeds left for host allocation. Separate search arm and small-V ablation remain distinct. Host must subsequently measure full-turn 2s/50ms and run 392 paired fixtures; do not invent host timing on Mac.

## Handoff

Not started.

## Log

- 2026-10-15 05:51Z codex-bigv@caspers-macbook: created
