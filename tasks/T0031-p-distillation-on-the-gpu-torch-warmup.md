---
id: T0031
title: "P distillation on the GPU: torch warmup/distill stages and T0011 cache reader"
status: open
owner: ""
needs: []
writes: [training/distill_policy.py, fusion/torch_qat.py, tests/test_distill_gpu.py]
depends: [T0030]
land: pr
branch: ""
updated: 2026-10-11
---

## Goal

T0030's training/distill_policy.py runs warmup and distill in NumPy on the CPU even with --backend torch (only qat uses torch), about 1 h per epoch per stage on 24M rows, and RowCache needs a row.npy that T0011's data/pf_policy_cache (CACHE.json + rec/y_*/weight/teams) does not have. Add a float Torch path for warmup/distill with the same loss as fusion.policy_distill.distill_loss (alpha*CE + (1-alpha)*tau^2*KL, smoothing, head weights), building batches on the device, and let RowCache read T0011's cache (row ids = positions, identity from CACHE.json digests). Code and toy tests only; the real teacher/student run is a separate task.

## Done when

Torch warmup/distill losses and gradients match the NumPy path on toy rows (CPU, float tolerance); --backend torch is honored or refused, never ignored; RowCache opens a T0011-format cache and its identity binds CACHE.json digests; core tests and tests.test_torch_backend pass; PR open.

## Handoff

Not started.

## Log

- 2026-10-11 19:17Z claude@caspers-macbook: created
