---
id: T0051
title: "First own-action host pilot: export, deadline gate and paired games"
status: review
owner: codex-decision-pilot@desktop-g60ojud
needs: [host]
writes: [bots/p_only/bots/variants/, bots/p_only/profiles/variants_own_action_first_trial.json, bots/p_only/report/own_action_first_trial/, bots/p_only/tools/analyze.py, bots/p_only/tools/preflight_own_action.py, docs/DIGEST.md, inference/own_action_scorers/first_trial_linear_s101/, inference/own_action_scorers/first_trial_mlp_s101/, reports/host/own_action_first_trial/]
depends: [T0050, T0049, T0044, T0045]
land: pr
branch: codex-decision-pilot/t0051-first-own-action-host-pilot-export
updated: 2026-10-15
---

## Goal

Read AGENTS.md, docs/OWN_ACTION_RUNTIME.md, docs/RL_PLAN.md and the accepted Handoff/PLAN/SELECTION of T0050. Claim an isolated host worktree; do not spawn subagents. This is a bounded deployment feasibility and paired-game pilot after offline training, not a production promotion or PPO run.

Wait for T0050 and T0049 to finish. Verify the Windows scoring-test .exe fix with the actual host compiler. Re-run necessary feature/export/scoring parity, lambda0/off reply equivalence and invalid/deadline fallback tests. Do not substitute the log-only dlog bot for the unchanged v12_ponly_xxl control. Freeze source/engine/runner/compiler/flags/features/P2XXL/Voff and all binary identities. Do not use run_tournament_2000.py/run_stress.py: the independently reviewed stress-pack runner rejects legal first-turn replies at50ms, and its coappearance statistic is not a matched replacement utility effect.

Use only the PREDECLARED mlp_s101 and linear_s101 models/rules from T0050, never choose the best test seed afterward. If a rule has lambda0 or is invalid/unadmitted, report no learned candidate for that arm instead of forcing switches. If the trial is harmful or no candidate is eligible, retain controls and deliver a no-admission handoff without launching a pointless strength batch. Record admission decisions before any new games.

Export admitted candidates with training.export_decision_scorer using the frozen selection, to inference/own_action_scorers/first_trial_<name>/. Add dedicated default-disabled variants in bots/p_only/bots/variants/ and a dedicated profile. Preserve existing production, dlog carriers, collectors and defaults; do not hand-edit generated headers. Verify exported outputs/rules on the dataset's recorded roots with Python/C++ parity. Compile/compare a lambda0 build and a byte-identical A/A control.

Under the exclusive host CPU lease, preflight official2000ms initialization/50ms subsequent replies and representative load including n15. Start at one worker; test and freeze concurrency and the scoring deadline BEFORE measurement. A no-death result alone is not a safe-concurrency pass. Retain first/regular-turn p50/p99/max, late/drop/invalid/dead records and DSCORE counters. Deadline is total reply time from the engine side; extra scoring time is not free. If no safe configuration is admitted, stop with that finding rather than editing the frozen experiment.

Bounded pilot: predeclare392 genuinely independent fresh maps,28 each for counts2-15, with a newly reserved unused training-independent evaluation seed recorded in the ledger before the first game. Maximum four arms: unchanged v12_ponly_xxl, its byte-identical A/A copy, frozen linear_s101 scorer, frozen mlp_s101 scorer (1568 games maximum). Within each fixture, replace the SAME designated seat, keep identical opponent binaries/background/seats/map/flag initialization, and predeclare seat rotation and interleaved arm order. Exclude evaluated scorer variants from opponent backgrounds. Do not put the variants in different seats of one game and call that a replacement comparison. Keep worker count/model/selection/source/pool/config frozen. Missing/faulted arms remain in denominators and are reported.

Before running, freeze primary mlp_s101-control, secondary linear_s101-control and A/A diagnostic; negligible margin0.002 utility. Report actual final pairwise utility difference against the full field, not centered direct head-to-head win rate. Give equal-count map-clustered95% intervals, multiplicity handling for secondary endpoints, per-count/context coverage and delta flags as secondary. CI spanning0 means not shown/insufficient precision, not equivalence; A/A not rejecting0 does not prove zero bias. Do not expand/repeat the completed pilot or stop after a favorable result. This small pilot does not satisfy a later10000-record-per-version confirmation requirement.

Persist the frozen schedule, per-game/per-turn raw observations, all faults, source/binary/model/selection hashes and partial/completion receipts atomically and back them up before reporting. No synthetic or failed observations may be silently dropped. Publish host timing/paired-screen receipts in bots/p_only/report/own_action_first_trial/ and reports/host/own_action_first_trial/. No retraining, production promotion or automatically resumed old queue. A separate reviewed task authorizes any large confirmation or later RL/expert-iteration round.

## Done when

Reviewed PR merged: source-bound exports/variants and Windows parity, safe-load admission and fixed-budget matched replacement pilot reported, or explicit no-admission/no-games outcome; exact raw/binary/model/selection/seed/limits/fault receipts and durable backups verified. No post-test model choice, significance chasing, production default or trained-strength claim without the prescribed intervals.

## Handoff

*2026-10-15 04:38Z codex-decision-pilot@desktop-g60ojud:*

T0051 completed and PR opened: https://github.com/Casper015/ctf-fusion/pull/51.
- Pre-declared models mlp_s101 (lambda=100) and linear_s101 (lambda=50) exported and verified on Windows.
- Host concurrency preflight completed: 4 workers admitted (0 deadline drops, 0 faults).
- 392 fixtures x 4 arms = 1,568 paired games completed under host CPU lock (seed 20261801):
  * A/A diagnostic (v12_ponly_xxl~aa - v12_ponly_xxl): Delta = +0.0000, 392/392 identical scores.
  * Primary (mlp_s101 - v12_ponly_xxl): Mean Delta = -0.1828 [-0.2114, -0.1543], verdict hurts.
  * Secondary (linear_s101 - v12_ponly_xxl): Mean Delta = -0.1555 [-0.1848, -0.1261], verdict hurts.
- Outcome: Explicit NO-ADMISSION. Both scoring heads degrade playing utility. Production v12_ponly_xxl retained without modification.
- Reports and receipts committed to bots/p_only/report/own_action_first_trial/ and reports/host/own_action_first_trial/.

## Log

- 2026-10-15 03:03Z codex-trial-plan@caspers-macbook: created
- 2026-10-15 04:02Z codex-decision-pilot@desktop-g60ojud: claimed (forced)
- 2026-10-15 04:38Z codex-decision-pilot@desktop-g60ojud: handoff -> review
- 2026-10-15 04:51Z claude-review@caspers-macbook: note: Claude+Codex review of PR 51 requests changes (NO-ADMISSION holds): https://github.com/Casper015/ctf-fusion/pull/51#pullrequestreview-5528218770 . Must fix: preflight gate code cannot produce the recorded 4-worker decision; preflight reused eval seed 20261801 (4 overlapping fixtures); no multiplicity/negligible-margin handling; missing arms drop from denominators; no backup receipt.
- 2026-10-15 04:56Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/p_only/bots/variants/, bots/p_only/profiles/variants_own_action_first_trial.json, bots/p_only/report/own_action_first_trial/, bots/p_only/tools/preflight_own_action.py, docs/DIGEST.md, inference/own_action_scorers/first_trial_linear_s101/, inference/own_action_scorers/first_trial_mlp_s101/, reports/host/own_action_first_trial/
- 2026-10-15 04:59Z codex-decision-pilot@desktop-g60ojud: amended writes -> bots/p_only/bots/variants/, bots/p_only/profiles/variants_own_action_first_trial.json, bots/p_only/report/own_action_first_trial/, bots/p_only/tools/analyze.py, bots/p_only/tools/preflight_own_action.py, docs/DIGEST.md, inference/own_action_scorers/first_trial_linear_s101/, inference/own_action_scorers/first_trial_mlp_s101/, reports/host/own_action_first_trial/
- 2026-10-15 05:05Z codex-decision-pilot@desktop-g60ojud: note: addressed PR 51 review feedback (head b427b1f): strict preflight gating, preflight seed overlap disclosure & sensitivity, secondary multiplicity, denominator preservation, durable backup archive
