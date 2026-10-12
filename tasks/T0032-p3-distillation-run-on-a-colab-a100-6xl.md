---
id: T0032
title: "P3 distillation run on Colab: 8XL teacher, XXL warmup/distill/QAT, three seeds"
status: review
owner: claude@caspers-macbook
needs: []
writes: [reports/models/p_distill/, tests/test_repo_layout.py]
depends: []
land: pr
branch: claude/t0032-p3-distillation-run-on-a-colab-a100-6xl
updated: 2026-10-12
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

*2026-10-12 09:45Z claude@caspers-macbook:*

Done and in review: PR https://github.com/Casper015/ctf-fusion/pull/34 (commit 1017dd8, report only, reports/models/p_distill/).

Results (validation only): distill minus supervised XXL p3 -0.0134 / -0.0113 / -0.0100 (mean -0.0116), all in the joint head; 8XL teacher 0.028 below XXL; mixed4_8 QAT from distill costs +0.032 to +0.036. RUNS.json and eval.json are byte copies of the Colab receipts; REPRODUCE.md embeds every script with SHA-256.

Weights, teacher shard and logs: Google Drive cft-fusion/t0032/ (teacher_8xl_p3, warmup/distill/qat_s101-103, targets_train/IDENTITY.json, live mirror; lost_runtime_1-3 are stopped runs, not results). Colab runtime disconnected and deleted 2026-10-12 ~09:40Z.

Not done / next:
- QAT of the supervised warmups (3 runs, ~1.2 h in parallel on an A100 High-RAM) to measure distill vs supervised after quantization.
- Export of a distilled student and host evaluation (near/far NLL with T0004's tools, then T0013-style equal-time games); nothing here shows playing strength.
- Standard-RAM Colab cannot run distill (61 GB random-access working set); use High-RAM.

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
- 2026-10-12 03:16Z claude@caspers-macbook: note: Runtime 2 (A100 80GB High-RAM) ended ~03:10Z when the runtime type was changed to standard RAM, 3 min into the parallel students (nothing finished; its mirror kept as Drive t0032/lost_runtime_2). Runtime 3: A100-SXM4-40GB, 83 GB RAM, Drive remounted by Casper. Restart 03:15Z with the same scripts (launcher 46cc43db, driver v3 682b061b, students 75a5065c) plus t0032_restart.sh (aa27a926): driver v3 restored teacher_8xl_p3 from Drive, recopies and verifies the cache, remakes the teacher targets, then hands over to the three parallel students.
- 2026-10-12 05:28Z claude@caspers-macbook: note: Runtime 3 (A100 40GB, 83 GB RAM): warmup_s101-103 done 05:00-05:01Z and backed up to Drive (best_val 2.08523 / 2.08424 / 2.08286, mean 2.08411; T0011 P3 XXL mean 2.08427). Distill stalled: train cache 42 GB + teacher shard 19 GB read at random by three students thrashed the page cache (GPU 0 %, 200 MB/s disk reads, no batch log line in 17 min). At Casper's request, stored to Drive (teacher and three warmups verified loadable there) and switched back to High-RAM at 05:27Z. The restart cell's handover now triggers on the first student stage the driver starts (warmups are restored from Drive and skipped).
- 2026-10-12 07:40Z claude@caspers-macbook: note: Runtime 4 (A100 80GB High-RAM) from 05:31Z: teacher and warmup_s101-103 restored from Drive, cache recopied and verified, targets remade, distill_s101-103 side by side from 05:51Z (618 s/epoch, GPU 97 %, no disk reads). Distill epoch-8 supervised val (best): 2.07186 / 2.07294 / 2.07286 vs warmup 2.08523 / 2.08424 / 2.08286, distill minus warmup -0.0134 / -0.0113 / -0.0100 (mean -0.0116; the teacher is 0.028 below the warmups). Gap per epoch narrowed from about -0.041 at epoch 1. Three seeds: per-seed results, no interval claimed. Calibration, then QAT, running.
- 2026-10-12 09:44Z claude@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/34 opened (report commit 1017dd8). QAT done 08:54Z: mixed4_8 integer-forward val 2.10465 / 2.10861 / 2.10461. Calibrated per head, distill minus warmup: joint -0.01396 / -0.01188 / -0.01062, act +0.00029 / +0.00071 / +0.00074, target ~0. Colab runtime disconnected and deleted at Casper's request.
- 2026-10-12 09:45Z claude@caspers-macbook: handoff -> review
- 2026-10-12 09:45Z claude@caspers-macbook: amended title -> P3 distillation run on Colab: 8XL teacher, XXL warmup/distill/QAT, three seeds
- 2026-10-12 09:46Z claude@caspers-macbook: amended writes -> reports/models/p_distill/, tests/test_repo_layout.py
- 2026-10-12 16:09Z codex-review@caspers-macbook: note: Report accepted at 91915d1223674640575fddfd861b861d7b44f824: https://github.com/Casper015/ctf-fusion/pull/34#pullrequestreview-5493227117. Ten completed runs, arithmetic, six local layout checks and exact-head CI verified. No strength/deployment claim; finish status after PR #34 merge.
