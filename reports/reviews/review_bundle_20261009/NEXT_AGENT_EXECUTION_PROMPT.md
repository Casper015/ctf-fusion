# Execution Prompt: Correct P/V Integration, Then Confirm Carrier and Capacity

This is a proposed task for a subsequent implementation agent. Publishing this document does not perform or approve deployment of the experimental variants. Read [current status](CURRENT_STATUS_AND_V_ABLATION_REVIEW.md) first, then the two reports linked by [the bundle index](README.md). Recheck the repository's current source before acting; findings are anchored to 0a0badc.

## Objective

Produce a reproducible comparison of C08 Omega_LIT and Player5 v12 with the trained P models and optional V. Select the strongest complete system under the official CPU-only 2-second first-reply and 50-ms later-reply limits. Separate model quality, consumer correctness, decisions, match strength, and runtime evidence. Do not assume that the original README carrier, largest network, largest P increment, or a borderline pilot interval identifies the winner.

## Phase 1: Repair correctness and establish identities

1. Isolate real observation history from counterfactual branches. In C08, `refineChecked()` calls `planTurn(next,false)`, which advances the persistent fusion Session; Runtime restore omits that Session. Keep the public root Session immutable during branch work and use branch-local contexts with an explicit parent state. Reversing branch order or changing branch count must not change root features, root policy output, or the next real observation's history.
2. Define each deployment adapter precisely. v12 currently consumes mixed movement marginals; C08 uses joint movement in part of sampling and marginals elsewhere. Neither consumes the complete conditional move/action/target policy. Compare movement-only variants first, then add joint/action/target variants with explicit identities. Check ground-flag firing overrides and voluntary drop/target coverage.
3. Recompute the actual carrier-specific legacy forecast and fit alpha/temperature only on tune data, where required. Do not reuse a joint-v2 gate to approve a product-of-marginals consumer or a different duel heuristic without evaluation. Preserve requested-move labels and legal support.
4. Correct V branch semantics. Use fully settled states with consistent turn/HP/flag/score/respawn/attack fields. Average V over realized branch states; do not use one arbitrary scenario or average feature vectors as a substitute. Use exact terminal pairwise utility. Instrument V path activation; C08's current frozen duel configuration does not execute its V refinement path.
5. Keep V disabled in ordinary P-only candidates. Its manifest approval is false; non-null pointers are not approval. Additive coefficients 25/35/40/50 are experimental consumer parameters and require separate tune/confirmation evidence.
6. Freeze content hashes for candidate source, included headers, binary, model weights, adapter, engine, opponents, schedule, and runner limits. Select explicit binaries and reject stale artifacts. Do not label a candidate name as a source hash.

Acceptance evidence: targeted branch-history regression; integer/native and incremental/full parity on real and stress states; exact-terminal checks; actual adapter distribution checks; candidate-target/rare-action coverage; and command-level control/null comparisons across 2–15 counts. A score-equality smoke test on two maps is insufficient.

## Phase 2: Make experiment records trustworthy

Preserve full match logs and requested orders. Save first reply, post-startup raw durations or mergeable histograms, max/late/dead/fault fields and reasons, V/P call counts, encoder time, useful samples/nodes, and hashes. Missing values stay missing. Do not average game p99s and call the result pooled p99. Instrument the integrated path before claiming zero heap allocation.

Fail incomplete paired fixtures rather than silently analyzing survivors. Infrastructure exclusions must be predetermined and symmetric; retain candidate faults/timeouts in the official outcome. Add A/A repetitions under the intended concurrent CPU load to measure deadline-sensitive search variability. Preserve exact matched map, chance seed, opponents, and seat for each substitution.

Rename paired W/T/L fields to improved/unchanged/degraded utility. Report actual game wins or first-place results separately. Correct the original carrier/V write-ups' unsupported significance, undefeated, all-seed-null, runtime, and reproducibility claims without deleting their source records.

## Phase 3: Fresh tuning comparison

Use a new output directory and fresh maps disjoint from the reviewed P/V pilots and their related repeats. The 0a0badc V fixtures overlap all 80 corresponding e11c45b P fixtures; they are exploration only.

Start with C08 and v12 control, P-M, and P-L after consumer repair. Add XL/XXL when their artifacts pass correctness and the actual CPU budget. Keep neural V off during P capacity selection. P and V may choose different sizes.

For V, freeze P and compare V off, a tune-fitted score/time or current-heuristic baseline, and the trained V hybrid. Tune additive coefficients and activation/routing on tune only. Include actual per-count V call coverage and branch-state/regret diagnostics.

Keep candidate actions/search settings/scenarios fixed for component diagnostics. Separately hold complete CPU time fixed for deployment comparisons, allowing measured node/sample differences. Record both absolute final-system utility and incremental P/V gain. Larger per-call quality can displace useful search work.

Evaluate integer deployment NLL, calibration, lethal-hit/collision/interception/flagshoot/drop/possession events, rare actions and target coverage. For V, evaluate early/middle/late states, close scores, branch ranking, and the declared continuation regime. Use a higher-budget reference evaluator with independent scenarios for selecting and measuring the reference best action.

The existing scripts can help, but their artifact identity, completeness, null handling, and timing shortcomings must be addressed first. Do not treat their current summaries as the final evaluator.

## Phase 4: Freeze and confirm

Choose architecture, P/V capacities, alpha, coefficients, routing, thresholds, and search budgets before final confirmation. Repeat finalists with at least three independently trained seeds; more matches for one checkpoint do not replace seed replication.

Each final integrated version requires at least 10,000 independent confirmation map groups over counts 2–15, initially at least 500 groups/count. Counts 9/11/13/14 served by general still count. Seat rotations, paired baseline replays, and turns do not multiply independent group count. Use new maps and reserved whole opponent lineages, with the official representative field and stronger duel tests.

Compute paired map-cluster uncertainty with declared count weights. Separate final pairwise utility from official Bradley-Terry ratings; do not convert NLL or utility gains directly into BT points. Do not tune a router on confirmation data or repeatedly stop at the first favorable ordinary 95% interval. A 10,000-group run can still be inconclusive for a small effect.

Report complete startup and later-turn p50/p95/p99/p99.9/max, late-game rates, faults, cache/tier/call coverage, and useful search work on the intended CPU under representative load. Retain the safest validated fallback. CPU inference remains independent of offline CUDA training.

## Deliverables

- Corrected integration and focused regression evidence, with source and dependency identities.
- Reproducible frozen fixture/build manifests and complete per-match records.
- Separate verdicts for artifact correctness, prediction, V/branch decisions, CPU runtime, and match strength.
- An updated evidence table distinguishing exploration, tune, OOD, and final confirmation.
- Representative CPU-only C++ candidates for external testing, with exact build/export hashes and a clear explanation of enabled P/V consumers.

If a comparison remains inconclusive, report it as inconclusive and retain the smaller or safer validated candidate. Do not claim a universally best carrier, independently useful V, or deadline approval from the current small pilot alone.
