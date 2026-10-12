---
id: T0011
title: "P feature ablation: train p2 and p3 at XXL on identical data, offline comparison, export"
status: done
owner: codex@desktop-g60ojud
needs: [cuda]
writes: [reports/models/p_features/, inference/models_pf_p2xxl_vm/, inference/models_pf_p3xxl_vm/]
depends: [T0009]
land: pr
branch: codex/t0011-p-feature-ablation-train-p2-and-p3-at
updated: 2026-10-12
---

## Goal

Offline half of the P feature ablation (docs/FEATURE_FAMILIES.md sections 3 and 8, profiles/p_features.json): policy family p2 against p3 at the same capacity, the only change being the combat block. On the CUDA host after T0009 is merged: (1) Rebuild layout-3 shards from the same raw matches with the committed dataset's options: python -m training.dataset --raw data/raw --out data/shards_l3 --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8 plus the --stride/--keep/--holdout-family/--holdout-mode/--split-by values recorded in data/shards/index.json 'options'. (2) Audit before touching any limit: python -m training.feature_audit --shards data/shards_l3 --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --split train --out reports/models/p_features/audit. (3) Train both with the committed XXL general run's arguments (inference/models_xxl/MANIFEST.json general.sources.policy.config.args: epochs, lr, batch, count weighting) and seed 1, under the GPU lock: python tools/repo/task.py lock gpu --task T0011 -- python -m training.train_policy --model general --shards data/shards_l3 --capacity XXL --policy-family p2 --backend torch --device cuda --seed 1 --out runs/pf_p2_xxl_s1 <those args>; the same with --policy-family p3 --out runs/pf_p3_xxl_s1. (4) python -m training.compare_policy --shards data/shards_l3 --split test --run p2=runs/pf_p2_xxl_s1 --run p3=runs/pf_p3_xxl_s1 --compare p3:p2 --out reports/models/p_features/test_s1 (stratified joint/act/target log loss, map-clustered intervals). (5) Export for the equal-time screen: python -m training.export_mixed --policy-run runs/pf_p2_xxl_s1 --value-tier m --out inference/models_pf_p2xxl_vm and the same for p3 -> inference/models_pf_p3xxl_vm. (6) If p3 is a finalist, train seeds 2 and 3 identically and report all three. Report inconclusive intervals as 'not shown'; better prediction alone is not stronger play. PR with the reports and the two model sets (each set includes a P XXL general model and V M).

## Done when

PR merged with reports/models/p_features/ (audit, test comparison, run configs and learning curves) and inference/models_pf_p2xxl_vm/, models_pf_p3xxl_vm/ exported by training.export_mixed; T0013 can build the screen variants.

## Handoff

*2026-10-12 05:20Z codex@desktop-g60ojud:*

Completed and merged PR #21 https://github.com/Casper015/ctf-fusion/pull/21; head 23014eb184f667567ca4ca17d190bb6048d2bf1a, merge 516d671be561a627e39b13291177a23962aca85b. Exact-head verification CI passed. All six training/calibration/comparison runs, exports, actual-weight parity and delivery checks complete; reports/models/p_features/RESULTS.md and CHECKS.md, user outputs/T0011. P3 improves joint NLL, not a playing-strength claim. T0013 model dependency is satisfied. Completed stages must not be restarted. User authorized task status reconciliation on 2026-10-12.

## Continuous monitoring requested by user

User explicitly requested ongoing monitoring to prevent another memory interruption. Thread heartbeat automation id t0011 is ACTIVE, every5 minutes in the current Codex conversation. It monitors process/log progress and memory, continues validation/report/export/PR work on completion, and safely diagnoses/restarts only a failed stage after verifying receipts. Normal unchanged status stays quiet.

Native Windows guard is also alive: ../../t0011_memory_guard.ps1, PID9872, polls every15 seconds while verified pipeline PID34248/creation identity survives. build/t0011/memory_guard_pid.json and memory_guard_state.json expose health; memory_guard.jsonl records every sample and API receipt. It applies a40 GiB resident working-set hard maximum (Windows SetProcessWorkingSetSizeEx, flags6) to verified direct Python child stages. Reverify all PID command lines/creation dates before operating; if pipeline PID changes, use the new verified PID when restarting guard. It neither kills/restarts training nor changes data, weights, CLI arguments or global OS/pagefile configuration. The source and resource-policy rationale are appended to REPRODUCE.md; MEMORY_GUARD.json records a snapshot. Working-set limits can slow IO; do not treat training times across the limit change as equal-resource evidence.

Limit was applied at04:51Z during P2 seed1 epoch3 after epochs1/2. Physical availability improved from about2 GiB to14.7 GiB, child working set now40 GiB, private commit about17.16 GiB. Epoch3 completed normally, train2.27193145, val2.27599874, seconds201.603, optimizer_steps70764. CPU continues advancing, epoch4 underway. The GPU/CPU original training lock chain is still alive. Memory caps do not guarantee survival of a host reboot; monitor actual health and preserve failure provenance.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-11 03:58Z codex@desktop-g60ojud: claimed
- 2026-10-11 04:07Z codex@desktop-g60ojud: note: User requested T0011 alongside T0022. Owned worktree prepared from current main. Verified the exact 21000 original raw matches (no missing/extra/duplicate IDs), 30049121 original rows and original sampling options. Preparing layout-3 shards per team count with all data retained, workers=8, CPU lock queued behind T0022 landing checks. p2/p3 XXL training will use identical original args (8 epochs, batch1024, lr0.001, cosine min_lr0.0001, patience2/plateau1, inv_teams, seed1, FP32), then offline comparison and exports; GPU is not yet training.
- 2026-10-11 04:25Z codex@desktop-g60ojud: handoff -> active
- 2026-10-11 04:38Z codex@desktop-g60ojud: note: Host reboot after RAM exhaustion during seed1 P2 dataset loading; no epoch/checkpoint was produced. Original concatenate/subset path reached 84.55 GB private commit on a 64 GB host. All 21000-match rebuild, full feature audit and exact 30049121-row legacy parity receipts survived. Resume will use an explicitly recorded, bounded per-shard policy-only NPY mmap cache and launcher within report reproduction, preserving every eligible row/order/dtype and all frozen training arguments; no core source edits or reduced-data comparison.
- 2026-10-11 04:47Z codex@desktop-g60ojud: handoff -> active
- 2026-10-11 04:53Z codex@desktop-g60ojud: handoff -> active
- 2026-10-11 05:10Z codex@desktop-g60ojud: note: P2 seed1 completed all 8 epochs (188704 optimizer steps): final train2.19677252, val2.21738021. The unchanged trainer is now in NumPy validation calibration/evaluation before checkpoint save; no model.npz/config.json yet, so do not treat the run as complete or restart it. CPU advances, guard and CPU/GPU locks remain healthy. At05:10Z working36.86 GiB, private25.97 GiB, available physical15.69 GiB and commit31.20 GiB; GPU idle during CPU calibration is expected. P3 follows automatically after the verified P2 artifact saves.
- 2026-10-11 05:25Z codex@desktop-g60ojud: note: P2 seed1 completed successfully, calibrated checkpoint/config saved and verified against frozen arguments/source/24153825 train+2981919 val rows. Checkpoint SHA256 5c99d94be5efe45b30a1b8c0f0b79e4414d0cb7b77bccabd46dfdeb127ddcbba exactly reproduces the original committed XXL general policy source checkpoint. This validates the controlled data/loading reproduction, including the memory cap. P3 seed1 began05:20:26Z (child PID45516) and completed epoch1; do not infer final gains from an early epoch. Automatic guard applied40 GiB to the new child, CPU progresses, physical availability about15 GiB. Pipeline and locks unchanged.
- 2026-10-11 05:50Z codex@desktop-g60ojud: note: P3 seed1 completed all8 epochs, final train2.06033609 and smoothed combined val2.08454825, optimizer_steps188704. Currently in unchanged CPU calibration/evaluation before artifact save; calibrated paired joint-NLL comparison remains pending, so no uplift/strength claim. Guard remains live and CPU advances: at05:50Z working39.53 GiB, private28.81 GiB, available physical14.35 GiB/commit28.34 GiB. Waiting for normal completion; do not restart.
- 2026-10-11 06:00Z codex@desktop-g60ojud: note: P3 seed1 calibration completed and artifacts saved at05:59Z, exit0. Verified identical P2/P3 hidden widths, data/rows/source/seed and every training argument except family/output. P3 model SHA256 b3f863a0e1810a02bdaa87543d5adf1185b2269d99bce99eaa92c816becd9def. Automatic paired map-clustered validation comparison is now running as verified child PID43156 under pipeline34248; guard9872 applied40 GiB and CPU advances. Available physical46.27 GiB/commit49.15 GiB at06:00Z. Await confidence intervals before finalist decision; no playing-strength claim.
- 2026-10-11 06:05Z codex@desktop-g60ojud: note: Seed1 paired validation finished:2981919 rows,2112 map clusters; P3-P2 joint-NLL mean-0.08838297,95%CI[-0.08959273,-0.08717321]. Prespecified finalist rule passed without using test, so pipeline automatically started P2 seed2 at06:02Z, verified child20100; both families seeds2/3 will follow unchanged. User already informed. Guard9872 remains healthy with40 GiB cap; CPU advances, available physical10.57 GiB/commit36.59 GiB at06:05Z. Offline prediction improvement is not yet a playing-strength result.
- 2026-10-11 06:35Z codex@desktop-g60ojud: note: P2 seed2 completed8 epochs/188704 optimizer steps: final smoothed combined train2.19432457,val2.21472096; unchanged CPU calibration/evaluation underway before model/config save. CPU advances and guard remains healthy at06:35Z: working37.94 GiB,private28.75 GiB,available physical13.88 GiB/commit26.63 GiB. Seed2 first epoch437.119s,second228.486s,then epochs4-8 about180s; memory policy unchanged, do not infer model capacity throughput differences from these timings. Wait for verified artifact save, no restart.
- 2026-10-12 05:20Z codex@desktop-g60ojud: done
