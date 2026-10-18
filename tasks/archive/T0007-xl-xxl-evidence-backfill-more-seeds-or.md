---
id: T0007
title: "XL/XXL evidence backfill: more seeds or per-epoch logs"
status: dropped
owner: ""
needs: [cuda]
writes: [reports/models/]
depends: []
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

The S-to-XXL trend rests on one seed, 8 epochs and no per-epoch logs (89ee436). Train 1-2 more seeds for L, XL and XXL with the identical configuration, or commit the per-epoch logs. Keep M separate (NumPy, not PyTorch), or retrain it on PyTorch. Spec: [bots/p_only/TASK.md](../../bots/p_only/TASK.md), "Evidence backfill". This does not block the screens.

## Done when

- New gate reports, or per-epoch logs, under `reports/models/<tier>/`.
- A note in each tier's SUMMARY.md saying whether XL and XXL were still improving at epoch 8.

## Handoff

*2026-10-18 19:37Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). GPU-only backlog not on the current plan; any GPU run must also stay out of host timing windows. Reopen as a new task if needed.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from bots/p_only/TASK.md and EXPERIMENTS.md
- 2026-10-18 19:37Z claude-fervent@cloud: dropped
