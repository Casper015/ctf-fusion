---
id: T0071
title: "Train 8XL count-specific opponent P teacher models and audit data sufficiency"
status: done
owner: claude-fervent@cloud
needs: [host, cuda, ai-test]
writes: [reports/models/count_specialists_8xl/, training/train_count_specialists_8xl.py, tests/test_repo_layout.py]
depends: []
land: pr
branch: antigravity/t0071-train-8xl-count-specific-opponent-p
updated: 2026-10-18
---

## Goal

Train 19.04M parameter 8XL opponent P teacher models across team counts 2..15 for eventual distillation back to XXL capacity, and audit data sufficiency across count-specific corpora.

## Done when

All 14 8XL models trained, calibrated, evaluated on held-out test data, and documented with data sufficiency and distillation readiness receipts in reports/models/count_specialists_8xl/REPORT.md.

## Handoff

*2026-10-18 19:38Z claude-fervent@cloud:*

Closed by claude-fervent@cloud at Casper's request (2026-10-18). PR 63 merged (2026-10-16) although its review requested changes (https://github.com/Casper015/ctf-fusion/pull/63); those were not addressed. Result as committed: the 8XL teachers have a higher test loss than the XXL specialists at every count 2-15. Report: reports/models/count_specialists_8xl/REPORT.md. Its follow-up T0074 is dropped.

## Log

- 2026-10-16 18:49Z antigravity@desktop-g60ojud: created
- 2026-10-16 18:50Z antigravity@desktop-g60ojud: claimed
- 2026-10-16 18:52Z antigravity@desktop-g60ojud: handoff -> review
- 2026-10-16 19:04Z claude-review@caspers-macbook: note: Review of PR 63 (deae8bb, Claude + Codex gpt-6.1-sol): changes requested, https://github.com/Casper015/ctf-fusion/pull/63 . Arithmetic and the 19,042,691 param count are right; 8XL test loss is worse than XXL on all 14 counts. Must fix: 8XL from random init vs warm-started XXL confounds capacity with prior data; best epoch is 8 for 13/14 counts (under-trained, not overfitting; 'overfits after epoch 1-2' is false); the overfit gap mixes objectives; data sufficiency never measured and three verdict rules disagree; no map-cluster intervals, one seed. Also: wrong scaling figures, throughput claim, training ran before the task was created, no checkpoint hashes.
- 2026-10-18 19:37Z claude-fervent@cloud: note: Casper (2026-10-18, Claude session): close or drop this task; it has sat in review/blocked for days. Taking it with claim --force on that approval to change its status.
- 2026-10-18 19:38Z claude-fervent@cloud: claimed (forced)
- 2026-10-18 19:38Z claude-fervent@cloud: done
