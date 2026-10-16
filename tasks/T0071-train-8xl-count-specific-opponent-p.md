---
id: T0071
title: "Train 8XL count-specific opponent P teacher models and audit data sufficiency"
status: active
owner: antigravity@desktop-g60ojud
needs: [host, cuda, ai-test]
writes: [reports/models/count_specialists_8xl/, training/train_count_specialists_8xl.py, tests/test_repo_layout.py]
depends: []
land: pr
branch: antigravity/t0071-train-8xl-count-specific-opponent-p
updated: 2026-10-16
---

## Goal

Train 19.04M parameter 8XL opponent P teacher models across team counts 2..15 for eventual distillation back to XXL capacity, and audit data sufficiency across count-specific corpora.

## Done when

All 14 8XL models trained, calibrated, evaluated on held-out test data, and documented with data sufficiency and distillation readiness receipts in reports/models/count_specialists_8xl/REPORT.md.

## Handoff

Not started.

## Log

- 2026-10-16 18:49Z antigravity@desktop-g60ojud: created
- 2026-10-16 18:50Z antigravity@desktop-g60ojud: claimed
