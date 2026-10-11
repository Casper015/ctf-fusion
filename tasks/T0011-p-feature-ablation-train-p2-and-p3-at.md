---
id: T0011
title: "P feature ablation: train p2 and p3 at XXL on identical data, offline comparison, export"
status: active
owner: codex@desktop-g60ojud
needs: [cuda]
writes: [reports/models/p_features/, inference/models_pf_p2xxl_vm/, inference/models_pf_p3xxl_vm/]
depends: [T0009]
land: pr
branch: codex/t0011-p-feature-ablation-train-p2-and-p3-at
updated: 2026-10-11
---

## Goal

Offline half of the P feature ablation (docs/FEATURE_FAMILIES.md sections 3 and 8, profiles/p_features.json): policy family p2 against p3 at the same capacity, the only change being the combat block. On the CUDA host after T0009 is merged: (1) Rebuild layout-3 shards from the same raw matches with the committed dataset's options: python -m training.dataset --raw data/raw --out data/shards_l3 --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --workers 8 plus the --stride/--keep/--holdout-family/--holdout-mode/--split-by values recorded in data/shards/index.json 'options'. (2) Audit before touching any limit: python -m training.feature_audit --shards data/shards_l3 --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --split train --out reports/models/p_features/audit. (3) Train both with the committed XXL general run's arguments (inference/models_xxl/MANIFEST.json general.sources.policy.config.args: epochs, lr, batch, count weighting) and seed 1, under the GPU lock: python tools/repo/task.py lock gpu --task T0011 -- python -m training.train_policy --model general --shards data/shards_l3 --capacity XXL --policy-family p2 --backend torch --device cuda --seed 1 --out runs/pf_p2_xxl_s1 <those args>; the same with --policy-family p3 --out runs/pf_p3_xxl_s1. (4) python -m training.compare_policy --shards data/shards_l3 --split test --run p2=runs/pf_p2_xxl_s1 --run p3=runs/pf_p3_xxl_s1 --compare p3:p2 --out reports/models/p_features/test_s1 (stratified joint/act/target log loss, map-clustered intervals). (5) Export for the equal-time screen: python -m training.export_mixed --policy-run runs/pf_p2_xxl_s1 --value-tier m --out inference/models_pf_p2xxl_vm and the same for p3 -> inference/models_pf_p3xxl_vm. (6) If p3 is a finalist, train seeds 2 and 3 identically and report all three. Report inconclusive intervals as 'not shown'; better prediction alone is not stronger play. PR with the reports and the two model sets (each set includes a P XXL general model and V M).

## Done when

PR merged with reports/models/p_features/ (audit, test comparison, run configs and learning curves) and inference/models_pf_p2xxl_vm/, models_pf_p3xxl_vm/ exported by training.export_mixed; T0013 can build the screen variants.

## Handoff

*2026-10-11 04:25Z codex@desktop-g60ojud:*

Running on DESKTOP-G60OJUD, owned worktree ../ctf-fusion.worktrees/T0011, branch codex/t0011-p-feature-ablation-train-p2-and-p3-at, frozen source9a78345172010a207fe89c54af12704b8cc3fddf. T0022 is complete/done (report7a307af49a); GPU queue read T0011 -> T0026 -> T0007 -> T0012 after T0010 data. Only T0011 is claimed. Exact original21000 raw files/30049121 rows verified against original index, no missing/extra/duplicate IDs. data/raw is a read-only-use local junction to the original raw_main_20261009. Rebuilding layout3 per count with all original options and workers8 under CPU lock (task process46044, since04:12Z); all counts2-15 rebuilt, now full train feature audit. Driver ../../t0011_prepare.py --original-index <original-shards>/index.json is resumable per complete count. Queued CPU-locked ../../t0011_after_prepare.py --original-shards <original-shards> waits for successful audit.json, checks every old feature/label/weight row using ../../t0011_data_parity.py, then runs ../../t0011_train.py under GPU lock. Both CPU and GPU are held during the training pipeline. Runtime env: AI_TEST_ROOT points to host engine; CTF_AGENT codex@desktop-g60ojud, CTF_CAN host,cuda,ai-test; Python3.14/PyTorch2.11.0+cu128; OPENBLAS/OMP/MKL_NUM_THREADS=8. Full matched P2/P3 XXL args copied from committed general source: epochs8,batch1024,lr0.001,cosine min_lr0.0001,patience2,plateau1,inv_teams,FP32/TF32off,seed1. No row cap. Prespecified finalist rule: calibrated validation overall joint-NLL map-clustered95%CI wholly below zero -> train both families again at seeds2,3; otherwise seed1 only. Test is not used for selection; exports use fixed seed1. Future logs build/t0011/pipeline.log and pipeline_state.json; outputs reports/models/p_features/, runs/pf_p{2,3}_xxl_s{1,2,3}, inference/models_pf_p{2,3}xxl_vm. Next: observe audit/parity/first GPU epoch; if failure inspect output and locks before resuming, never duplicate a running job. After pipeline: validate compact reports and export identities, commit only declared reports/model lanes, push/open required PR, attach it, await checks/review and finish task through protocol. No policy/default deployment change or host strength claim. Transient driver sources must be embedded verbatim in a reproducibility document under the report lane before PR; generated results are not hand-edited.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-11 03:58Z codex@desktop-g60ojud: claimed
- 2026-10-11 04:07Z codex@desktop-g60ojud: note: User requested T0011 alongside T0022. Owned worktree prepared from current main. Verified the exact 21000 original raw matches (no missing/extra/duplicate IDs), 30049121 original rows and original sampling options. Preparing layout-3 shards per team count with all data retained, workers=8, CPU lock queued behind T0022 landing checks. p2/p3 XXL training will use identical original args (8 epochs, batch1024, lr0.001, cosine min_lr0.0001, patience2/plateau1, inv_teams, seed1, FP32), then offline comparison and exports; GPU is not yet training.
- 2026-10-11 04:25Z codex@desktop-g60ojud: handoff -> active
