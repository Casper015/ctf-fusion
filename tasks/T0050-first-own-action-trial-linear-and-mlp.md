---
id: T0050
title: "First own-action trial: linear and MLP training with frozen evaluation"
status: review
owner: codex-decision-pilot@desktop-g60ojud
needs: [host]
writes: [reports/models/own_action_first_trial/]
depends: [T0038]
land: pr
branch: codex-decision-pilot/t0050-first-own-action-trial-linear-and-mlp
updated: 2026-10-15
---

## Goal

Read current AGENTS.md, CONTRIBUTING.md, docs/RL_PLAN.md and docs/OWN_ACTION_RUNTIME.md. Claim this task in an isolated worktree. Do not spawn subagents. Casper authorized the first small decision-learning experiment. This is offline own-action gain learning and constrained policy improvement, not PPO or self-play. All repository prose is English.

Use ONLY the accepted T0038 host corpus/dataset from PR46, seed20261701: data/decision/host_pilot_dataset, ctf-fusion-decision-2, 336 maps, train/val/test268/37/31 maps, 1317 candidates, 5268 continuations. Verify index/NPZ/source/carrier hashes against DATA_AUDIT, FINAL_ACCEPTANCE and CHECKS before starting. Keep P2XXL/v12_ponly_xxl_dlog/Voff fixed. Do not mix Colab data, alter splits, relabel roots, recollect or replay successful games, retrain P/V, modify production defaults, or use the independent stress-pack tournament as training evidence.

Before any training or test read, commit a PLAN.json with exact source/dataset/model/config identities and the following bounded plan. Six runs only: linear and MLP(hidden64,16), each seeds101,102,103. Primary loss gain; epochs50 maximum, patience5, batch256, lr1e-3, weight decay1e-4, existing root/count/noise weighting unchanged. Use the existing NumPy CPU trainer under the host CPU lease. Seed numbers here are optimizer RNG seeds, not new game/map seeds. Predeclare primary model mlp_s101 versus incumbent; linear_s101 is the main secondary comparator; other seeds are reproducibility/sensitivity comparisons, not extra independent data. No pairwise-loss/hyperparameter/architecture sweep in this task.

Example commands for seed101 (repeat with seeds102 and103 and distinct matching output names; all six planned runs only):

```bash
python -m training.train_decision_ranker --data data/decision/host_pilot_dataset --model linear --loss gain --seed 101 --epochs 50 --patience 5 --batch 256 --lr 1e-3 --weight-decay 1e-4 --out runs/own_action_first_trial/linear_s101
python -m training.train_decision_ranker --data data/decision/host_pilot_dataset --model mlp --hidden 64,16 --loss gain --seed 101 --epochs 50 --patience 5 --batch 256 --lr 1e-3 --weight-decay 1e-4 --out runs/own_action_first_trial/mlp_s101
```

Run these inside the host CPU lease, after PLAN.json is committed. Do not pass --exploratory or bypass identity failures. Use train only for optimization and val only for early stopping/rule selection. Report loss, candidate coverage, switches, per-count/context behavior, run-to-run variation and overfit symptoms; current validation has only37 independent maps.

Freeze the incumbent-anchored zscore rule with training.evaluate_decisions select, all six named runs, primary mlp_s101, existing lambda grid including0, and negligible margin0.002 utility. The six names must not contain the comparison delimiter ' - '. Record the resulting SELECTION.json and all model/config/dataset/feature/carrier hashes before opening test. Evaluate that frozen selection ONCE through training.evaluate_decisions test. Preserve its secondary-comparison correction. Do not tune, select a different primary/seed, retrain or expand the test set after inspecting31 test maps. Original headroom+0.0525/+0.0895 is diagnostic only, not achieved model gain.

Persist per-run config/log/checkpoint receipts and failure/completion state atomically; never rerun a verified completed run merely to improve a result. If genuine interruption occurs, preserve it and restore the same frozen run/config/RNG state where supported, otherwise report the restart as a new attempt. Raw datasets/checkpoints stay ignored, durably backed up with hashes. Publish English report, PLAN, selection, offline result, reproducibility commands and run receipts under reports/models/own_action_first_trial/. Do not export deployment weights or run matches here. End with benefit/harm/negligible/insufficient-precision per the existing evaluator, with a bounded recommendation for the host pilot. A lambda0 choice means no learned override; never claim it as a learned gain.

## Done when

Reviewed PR merged: bounded six-run trial and one frozen test evaluation completed or truthful failure/no-admission reported; exact accepted dataset and independent-map counts preserved; all plans/selection/results/checkpoint receipts pinned and backed up; no test-driven tuning, new games, P/V retraining, deployment/default change or strength claim. Handoff identifies mlp_s101 and linear_s101 selections (including lambda0), run paths/hashes and whether either is eligible for the separate host pilot.

## Handoff

*2026-10-15 03:17Z codex-decision-pilot@desktop-g60ojud:*

PR opened: branch codex-decision-pilot/t0050-first-own-action-trial-linear-and-mlp (head e371c0f). Bounded 6-run trial and single frozen test evaluation completed under host CPU lease strictly within writes lane reports/models/own_action_first_trial/. Accepted T0038 dataset verified (336 maps, train/val/test 268/37/31 maps). Pre-declared primary mlp_s101 (lambda 100.0) and secondary linear_s101 (lambda 50.0) frozen in SELECTION.json (sha256 a571b05a4473eaaa). Test evaluation on 31 maps: primary mlp_s101 - incumbent mean -0.0151 [-0.0563, +0.0261] (insufficient precision); secondary linear_s101 - incumbent mean -0.0203 [-0.0846, +0.0440] (insufficient precision); test headroom +0.0487. All receipts, report and plans committed. Both models eligible for T0051 deployment feasibility/preflight with caution under matched replacement protocol.

## Log

- 2026-10-15 03:03Z codex-trial-plan@caspers-macbook: created
- 2026-10-15 03:04Z codex-trial-plan@caspers-macbook: amended goal
- 2026-10-15 03:10Z codex-decision-pilot@desktop-g60ojud: claimed
- 2026-10-15 03:17Z codex-decision-pilot@desktop-g60ojud: handoff -> review
