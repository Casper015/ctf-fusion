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

Run T0030's teacher/student plan (proposed follow-up B) for the P3 finalist of T0011, on a Colab A100 (Casper's request, 2026-10-11), with T0031's code (branch claude/t0031-p-distillation-on-the-gpu-torch-warmup, commit 39eb4e5, PR #24). Offline training and validation only: no test split, no export to inference/, no timing, no games.

Data: T0011's data/pf_policy_cache (24,153,825 train and 2,981,919 val rows), copied from Casper's Drive and bound to reports/models/p_features/POLICY_CACHE.json (every array rehashed with `distill_policy verify-cache`).

Frozen settings: T0011's arguments for every stage (8 epochs, batch 1024, cosine lr 1e-3 -> 1e-4, weight decay 1e-5, patience 2, plateau decay 1, smoothing 0.01, act/target weights 0.5, count weighting inv_teams, FP32, no TF32). Family p3.

1. Teacher: 8XL p3 (mover 2048, squad 8192-2048, act 2048, target 256; 20.95M parameters, about 3x the 6XL compute; Casper asked for a larger teacher on the 40 GB A100) with the unchanged training/train_policy.py on the cache (launcher binds the cache to POLICY_CACHE.json instead of shards_l3), training seed 100, calibrated on val. Offline only, never deployed.
2. Teacher targets on the train split at tau 1 (`distill_policy teacher-targets --device cuda`).
3. Students XXL p3, training seeds 101, 102, 103, each two paired arms from scratch with `distill_policy train --backend torch --val-cache`: warmup (alpha 1, the supervised control through the same code path) and distill (alpha 0.5, tau 1). Then QAT mixed4_8 from each distill model (2 epochs, lr 1e-4 -> 1e-5).
4. Report under reports/models/p_distill/: per seed and arm the supervised val loss (best, calibrated per head), paired distill-minus-warmup differences over seeds, QAT val loss on the deployed mixed4_8 arithmetic, the teacher's val loss, and T0011's P3 XXL seeds 1-3 as the reference. Three seeds give no interval worth claiming; report each seed and say so.

## Done when

Teacher, targets, 3x(warmup, distill, qat) completed with config/log/receipt per run; report in reports/models/p_distill/ with the commands, data binding, per-seed results and limits; no test split used; PR open.

## Handoff

Not started.

## Log

- 2026-10-11 19:45Z claude@caspers-macbook: created
- 2026-10-11 19:45Z claude@caspers-macbook: claimed
- 2026-10-11 19:51Z claude@caspers-macbook: note: Started 19:50Z on Colab A100-SXM4-40GB, code 3858b0f (T0031 branch). Train cache copied from Drive and verified (identity 350264d6881ddf31, 24,153,825 rows; val 20de39893d174c1b). Driver reports/models/p_distill/scripts/t0032_run.sh (sha cefaccd9...), launcher t0032_train_cache.py (sha 46cc43db...). Stage 1: 6XL p3 teacher, seed 100. Finished stages are copied to Drive cft-fusion/t0032/.
- 2026-10-11 19:58Z claude@caspers-macbook: amended goal
- 2026-10-11 19:58Z claude@caspers-macbook: note: Teacher changed 6XL -> 8XL at Casper's request (use the A100's room). 6XL p3 was stopped after epoch 1 (317 s/epoch; train 2.22714, val 2.18051; log kept as stopped_teacher_6xl_p3.log, not a result). Restarted 19:57:39Z with code 39eb4e5 (GPU teacher targets) and driver sha 1cea884a.
- 2026-10-11 23:18Z claude@caspers-macbook: note: Runtime 1 (A100 40GB, driver v2) ended between 20:47Z and ~21:15Z with nothing finished: its mirror on Drive (moved to cft-fusion/t0032/lost_runtime_1) has the 8XL teacher's epochs 1-6 (val 2.18051->2.07133 for epoch 6; log only, no model.npz), not a result. Restarted 23:17Z on Colab Pro, new Drive notebook T0032_p3_distill_colab_pro.ipynb, A100-SXM4-80GB High-RAM, driver v3 reports/models/p_distill/scripts/t0032_colab_pro.sh (sha c2d20f03...), launcher unchanged (46cc43db...). Code main 95e173b (PR #24 merged; training/ and fusion/ trees equal 39eb4e5's, checked by the driver). v3 restores finished stages from Drive, rehashes the cache (verify-cache), and mirrors /content/t0032 to Drive t0032/live every 5 min. Same plan and seeds (teacher 100, students 101-103).
- 2026-10-12 01:44Z claude@caspers-macbook: note: Driver v3 bug, fixed before any training: POLICY_CACHE.json's identity block is the same for train and val, so the first v3 (sha c2d20f03) chose the train folder as the val source; stopped 23:32Z, val copy discarded, train copy (verified) kept. v3 sha 682b061b matches splits by row count and every array's data_sha256. Cache identities: train 350264d6881ddf31, val d899cf9eac2d2bf5 (7 arrays incl. teams.npy; runtime 1 logged val 20de39893d174c1b, which recomputes as the same split without teams.npy). 8XL p3 teacher (seed 100) trained 23:36-00:34Z on A100-SXM4-80GB: epoch-8 val 2.05652 vs T0011 P3 XXL seeds 1-3 mean 2.08427 (range 0.00112); epochs 1-6 equal runtime 1's to all printed digits. Its calibration runs in NumPy on the CPU (unchanged trainer): collect pass ~1 h, temperature fits at 01:42Z, evaluation pass next; finish expected ~02:50Z. Colab Pro: 171.9 compute units at the start, ~6.8/h on this A100.
- 2026-10-12 02:47Z claude@caspers-macbook: note: Teacher done 02:47:22Z (calibration on the CPU took 2h13m) and backed up to Drive t0032/teacher_8xl_p3. Calibrated val (smoothing 0, inv_teams, as T0011 logs): joint 1.83361 / act 0.24144 / target 0.11351, beta 1.0307/1.0016/1.0025; T0011 P3 XXL seeds 1-3 mean 1.86159 / 0.24223 / 0.11352. Teacher targets started on the GPU at 02:47:22Z.
- 2026-10-12 03:09Z claude@caspers-macbook: note: Teacher targets done 02:47:22-02:52:00Z on the GPU (shard row_digest 62279b07ba8689b8). Driver v3 then ran warmup_s101 alone: 311 s/epoch with the A100 at ~26 % (each trainer is one CPU core); epochs 1-2 val 2.19849 / 2.16580 vs T0011 P3 XXL mean 2.19443 / 2.16531. Stopped at 03:07Z during epoch 3 (kept as stopped_warmup_s101, not a result); reports/models/p_distill/scripts/t0032_students.sh (sha 75a5065c) now runs seeds 101, 102, 103 side by side with the driver's commands (warmup -> distill -> qat each), 4 BLAS threads per seed. GPU 100 % with three students. Timings are not evidence; results do not depend on the overlap.
