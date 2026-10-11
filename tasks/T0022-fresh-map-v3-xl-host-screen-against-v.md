---
id: T0022
title: "Fresh-map V3-XL host screen against V off and V2-M"
status: open
owner: ""
needs: [host, ai-test]
writes: [profiles/screen_v3xl_fresh.json, bots/p_only/report/v3xl_fresh/]
depends: [T0009]
land: direct
branch: ""
updated: 2026-10-11
---

## Goal

Measure whether the already trained designated-seat V3-XL improves the complete C08 bot over V off and the existing V2-M on unseen maps, before running the separate PR #10 search experiment. This is a host screening task, not retraining or final production confirmation. PR #9 adds infrastructure, not a new checkpoint; identify the actual model being tested. Keep T0010/T0012/T0014 and the T0016 5XL/student capacity study separate.

1. Freeze the experiment before play. Start an owned worktree from current main and record its commit, compiler/flags, engine, opponent binaries, source/model manifests, and all candidate bundle/binary SHA-256 hashes. Use existing arms c08_ponly_pxxl_v3xl (V off), c08_ponly_pxxl_vm_v (old V2-M), and c08_ponly_pxxl_v3xl_v (new V3-XL), plus an identical-binary A/A copy of the new V arm. Prove every policy array is the same frozen p2 XXL model. Verify family-carrier versus legacy-carrier equivalence with V off, and explicitly verify PR #10 portfolio and per-branch ply-2 settings are off. Keep depth, coefficients (refine 50, beam 35), search budgets, V call/deadline settings, and all non-V behavior fixed and recorded. The old/new V comparison is a deployed-model comparison: their features, capacities and training data differ, so it does not isolate feature quality or capacity. If an existing wrapper cannot satisfy the controls, report the failed preflight and create a separate code task rather than touching another owner's lanes.

2. Reserve seed 20261230 here for this screen only. At execution, refetch the board and audit tasks/profiles/recorded runs for conflicts; if another run has consumed it, reserve a different unused seed in the task log before play. Do not reuse 20261108 (known tuning maps), 20261220 (T0016 consumed), or seeds reserved by other tasks. Never train, calibrate or tune coefficients/budgets on these screen maps.

3. Add profiles/screen_v3xl_fresh.json using bots/p_only/profiles/screen_v3xl_matched.json as the starting schema: seed 20261230; team counts 2 through 15; 50 maps per count; the same frozen 14-opponent reference roster; 4 workers fixed for the whole run; first_ms 2000 and turn_ms 50; keep_matches true; the three arms above plus aa=[c08_ponly_pxxl_v3xl_v]; the same three paired comparisons. This is 700 independent map fixtures and 2,800 games. Set work_dir=data/p_only/v3xl_fresh and report_dir=bots/p_only/report/v3xl_fresh. Include all model dependencies explicitly in the evidence, including pre-included/shared value headers; record actual checkpoint/export identity and calibration.

4. Run the repository core and P-only contract suites before measurement. Build/preflight/dry-run under the host CPU lock and record readiness; run the official screen under the same task's CPU lock. Lock options must precede the positional lock name: python tools/repo/task.py lock --task TASK_ID cpu -- python bots/p_only/tools/run_screen.py --config profiles/screen_v3xl_fresh.json --dry-run, then the identical command without --dry-run. Set AI_TEST_ROOT on the host, never commit an absolute host path. Prebuild reference opponents before measured jobs. Interleave paired arms with rotated order as the runner does. Do not pull, rebuild, clean build/, train or run competing benchmarks during measurement. Resume only with unchanged identities and schedule; reject stale/duplicate results rather than silently pooling them. Plumbing smoke runs are not strength evidence.

5. Primary outcome: equally count-weighted paired utility V3-XL minus matched V off, with 95% intervals clustered by independent map. Secondary: V3-XL minus V2-M and V2-M minus V off. Report all counts and the 2, 3-5, 6-10 and 11-15 bands, but do not promote a favourable subgroup when the overall comparison is inconclusive. Report A/A paired differences and intervals next to the primary result; examine effect size against rerun noise. Utility is the share of rivals beaten with half credit for ties, not a game win rate. Missing fixtures, incompatible identity, faults or delays must remain visible in denominators and reporting; do not selectively discard losing/faulted games. A paired interval crossing zero means 'not shown'. A wholly positive interval is a screening signal, not production approval or a fresh multi-seed confirmation.

6. Report whole-reply first-turn and subsequent-turn p50/p99/max, faults, timeouts and late replies at the official limits. Inspect VALUE/VBUDGET and SEARCH counters for V calls/time, changed choices, skipped calls and fallback decisions, samples displaced and truncations. Keep absent counters missing, especially the two-team path; show covered-game/turn denominators. Record actual host load, worker count, lock receipt, command lines, elapsed time, engine and binary hashes, and completeness. Recheck binary/source hashes after the run. Do not label deadline-before-call checks as interruptible inference.

7. Commit only the profile and generated compact reports under bots/p_only/report/v3xl_fresh/; raw results, match trajectories, checkpoints and binaries remain ignored. Use the existing analyzer for generated SCREEN.md/JSON, and add a reproducible supplemental analysis only if needed to establish valid map clustering, A/A noise or counter coverage. Document how to rerun any supplement; never hand-edit generated results. Handoff states shown/not shown/harm for every comparison and the limits of this single checkpoint/seed screen. Keep production defaults unchanged; a promising result requires independent confirmation through T0006 with frozen settings and independently trained finalists.

## Done when

The fresh-seed profile and complete generated report are committed and landed through the task protocol. All 700 paired fixtures (2,800 games including A/A) are accounted for, with paired map-clustered intervals, subgroup results, A/A noise, actual model/P identities, full-reply timing, faults, VALUE/VBUDGET/SEARCH counter coverage, exact commands, host/load/lock and frozen hashes. Preflight contract checks pass. The Handoff distinguishes an observed screening gain from an inconclusive result and final confirmation, names the tested checkpoint, records consumed seed 20261230 (or its logged replacement), and preserves production defaults. An interrupted or failed run remains explicitly incomplete and is handed off with exact recovery steps; it is not marked done.

## Handoff

Not started.

## Log

- 2026-10-11 00:01Z codex@desktop-g60ojud: created
