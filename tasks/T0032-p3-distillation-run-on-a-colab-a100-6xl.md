---
id: T0032
title: "P3 distillation run on a Colab A100: 6XL teacher, XXL warmup/distill/QAT, three seeds"
status: active
owner: claude@caspers-macbook
needs: []
writes: [reports/models/p_distill/]
depends: []
land: pr
branch: claude/t0032-p3-distillation-run-on-a-colab-a100-6xl
updated: 2026-10-11
---

## Goal

@/private/tmp/claude-501/-Users-casper-code-ctf-fusion--claude-worktrees-pr22-colab-mcp-61c8cf/8273d059-efaf-4428-877e-d659a41051bd/scratchpad/goal32.md

## Done when

Teacher, targets, 3x(warmup, distill, qat) completed with config/log/receipt per run; report in reports/models/p_distill/ with the commands, data binding, per-seed results and limits; no test split used; PR open.

## Handoff

Not started.

## Log

- 2026-10-11 19:45Z claude@caspers-macbook: created
- 2026-10-11 19:45Z claude@caspers-macbook: claimed
- 2026-10-11 19:51Z claude@caspers-macbook: note: Started 19:50Z on Colab A100-SXM4-40GB, code 3858b0f (T0031 branch). Train cache copied from Drive and verified (identity 350264d6881ddf31, 24,153,825 rows; val 20de39893d174c1b). Driver reports/models/p_distill/scripts/t0032_run.sh (sha cefaccd9...), launcher t0032_train_cache.py (sha 46cc43db...). Stage 1: 6XL p3 teacher, seed 100. Finished stages are copied to Drive cft-fusion/t0032/.
