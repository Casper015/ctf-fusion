---
id: T0005
title: "P-only phase 4: four-turn lookahead"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/]
depends: [T0003]
land: pr
branch: ""
updated: 2026-10-10
---

## Goal

Extend C08 P-only from two to four simulated turns with the same single P, each state predicted from its own simulated parent. Measure the current shared-prediction approximation first. Spec: [bots/p_only/TASK.md](../bots/p_only/TASK.md), Phase 4.

## Done when

- Depth is a setting from 2 to 4. The contract test proves that branch order and count never change the real root, and keeps the engine comparison.
- Fixed-sample and fixed-time screens against the 4-turn scripted control and the 2-turn P-only arm, committed under `bots/p_only/report/`.
- The bot changes land through a pull request with that evidence.

## Handoff

Not started. This task writes all of `bots/p_only/`, so it cannot be active while a P-only screen runs. That is intended: the screen's bots must not change under it. The code can be written on any machine; split off a code-only task if that helps.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
