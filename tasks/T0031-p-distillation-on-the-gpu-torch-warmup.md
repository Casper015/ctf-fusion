---
id: T0031
title: "P distillation on the GPU: torch warmup/distill stages and T0011 cache reader"
status: review
owner: claude@caspers-macbook
needs: []
writes: [docs/P_MIXED_PRECISION.md, fusion/torch_qat.py, tests/test_distill_gpu.py, tests/test_precision.py, tests/test_torch_backend.py, training/distill_policy.py]
depends: []
land: pr
branch: claude/t0031-p-distillation-on-the-gpu-torch-warmup
updated: 2026-10-11
---

## Goal

T0030's training/distill_policy.py runs warmup and distill in NumPy on the CPU even with --backend torch (only qat uses torch), about 1 h per epoch per stage on 24M rows, and RowCache needs a row.npy that T0011's data/pf_policy_cache (CACHE.json + rec/y_*/weight/teams) does not have. Add a float Torch path for warmup/distill with the same loss as fusion.policy_distill.distill_loss (alpha*CE + (1-alpha)*tau^2*KL, smoothing, head weights), building batches on the device, and let RowCache read T0011's cache (row ids = positions, identity from CACHE.json digests). Code and toy tests only; the real teacher/student run is a separate task.

## Done when

Torch warmup/distill losses and gradients match the NumPy path on toy rows (CPU, float tolerance); --backend torch is honored or refused, never ignored; RowCache opens a T0011-format cache and its identity binds CACHE.json digests; core tests and tests.test_torch_backend pass; PR open.

## Handoff

*2026-10-11 19:23Z claude@caspers-macbook:*

In review: PR https://github.com/Casper015/ctf-fusion/pull/24 (branch claude/t0031-p-distillation-on-the-gpu-torch-warmup). Torch warmup/distill (fusion.torch_qat.FloatPolicy + device_batch through qat_loss) and RowCache for T0011's data/pf_policy_cache (positions as row ids, identity from CACHE.json digests, verify-cache). Tests: tests.test_torch_backend + tests.test_precision 44 OK with torch 2.11 CPU on Colab; core discover OK on the Mac (41 skipped). Not done: no real training. The real run (6XL teacher, then XXL warmup/distill/QAT, 3 seeds) needs (1) data/pf_policy_cache/train (~45 GB; only val, 5.55 GB, was shared on Drive and its rec digest cd55cbdc matches POLICY_CACHE.json), (2) runs/pf_p2_xxl_s1 and runs/pf_p3_xxl_s1 float checkpoints (only INT16 exports are in git), (3) a GPU runtime; Colab's A100 session ended and it fell back to CPU. teacher-targets still runs in NumPy (~2.6 h projected for 24M rows).

## Log

- 2026-10-11 19:17Z claude@caspers-macbook: created
- 2026-10-11 19:17Z claude@caspers-macbook: amended depends -> none
- 2026-10-11 19:17Z claude@caspers-macbook: note: Depends on T0030's code, which is on main (PR #22 merged 4368fd1); T0030 itself is still 'review' under its owner, so the board dependency was dropped instead of forcing it.
- 2026-10-11 19:17Z claude@caspers-macbook: claimed
- 2026-10-11 19:19Z claude@caspers-macbook: amended writes -> docs/P_MIXED_PRECISION.md, fusion/torch_qat.py, tests/test_distill_gpu.py, tests/test_precision.py, tests/test_torch_backend.py, training/distill_policy.py
- 2026-10-11 19:22Z claude@caspers-macbook: note: PR: https://github.com/Casper015/ctf-fusion/pull/24 (Torch warmup/distill, T0011 cache reader; 44 torch+precision tests OK on Colab CPU). Real run still needs the 45 GB train split and runs/pf_p{2,3}_xxl_s1 float checkpoints from the host; only val (Drive folder 1dfyW9Nx...) was provided.
- 2026-10-11 19:23Z claude@caspers-macbook: handoff -> review
- 2026-10-11 19:28Z claude@caspers-macbook: note: Colab A100-SXM4-40GB (12 vCPU, 83 GB RAM): val split from Drive verified against CACHE.json (identity 20de39893d174c1b). Throughput only (real train() path, 51,200 val rows, random weights, nothing kept; not evidence): 6XL teacher warmup 0.19 h/epoch of 24.15M rows, XXL warmup 0.09, distill 0.14, QAT 0.31, NumPy teacher-targets 1.0 h. Still missing: train split and runs/pf_p{2,3}_xxl_s1.
