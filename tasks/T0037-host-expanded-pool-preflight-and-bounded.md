---
id: T0037
title: "Expanded-pool preflight and bounded decision data pilot (run on Colab by Casper's decision, not host evidence)"
status: active
owner: claude-ci@caspers-macbook
needs: []
writes: [bots/p_only/report/decision_collection/, docs/DIGEST.md]
depends: [T0035, T0036, T0039, T0040]
land: pr
branch: claude-ci/t0037-host-expanded-pool-preflight-and-bounded
updated: 2026-10-12
---

## Goal

Read AGENTS.md and CONTRIBUTING.md, claim this task and use its isolated worktree. Respect active/review write lanes and machine locks. All repository prose is English. Do not spawn subagents. Preserve completed experiments and consumed seeds. Do not reopen T0012/T0014, change production defaults, or interfere with T0032. Run the bounded host preflight and fresh-map decision-data pilot after T0035 and T0036 are merged; this task authorizes the pilot, not model training or a strength screen. All formal host work uses exclusive CPU leases; no concurrent Windows training or competing benchmarks. Colab T0032 is separate.

Preflight the frozen expanded pool: build all available requested variants, retain the five missing-weight A-E records with zero sampling, prove all source/transitive/model/calibration/binary identities, discover binary aliases and recheck reserved-lineage exclusion. Use common recorded probes and A/A checks to establish neural-group response differences/stability. Measure the complete official 2000/50ms behavior under representative count/load, including n15. Start at one worker; choose and freeze concurrency from preflight, not during collection. Runtime failures remain in an admission/quarantine ledger and change the effective pool identity; no silent skips or substitutions. Simulate the exact sampler per count and publish actual component/group/lineage masses and cap limitations.

Freeze the admitted v12 P2 incumbent, V off, models, search settings and budget. Do not wait for T0032 merely to collect this baseline; if a future distilled model is chosen instead, it must pass its separate deployment admission and gets separate data provenance. Reserve a genuinely unused training-only seed after inspecting the ledger, actual schedules and host receipts, and record it before the first game. No seed is selected by this task specification.

Collect a predeclared 300-500 independent-map pilot across counts 2-15, balanced as closely as possible (e.g. 336 or 490 total maps). Keep complete matches and all-seat replies/faults. Select one representative root per map using a deterministic observable-state-only rule; cover carrier/flag opportunities, contact, interception/defense and quiet movement. Record stratum opportunities and selection rates; never pick roots based on final outcomes. Retain root shortages/drops and actual K. For each root use the incumbent plus three actual logged shortlist candidates where legal; initially four continuations per candidate. Reuse common controlled randomness where feasible, record repeat/prefix/internal-state provenance, and keep opponents live after the root. Do not freeze the future enemy reply tape to eliminate noise.

Preserve all per-repeat scores/outcomes, source receipts, official limits and flags. Predeclare any replication expansion/stop rule; never stop as soon as a favorable label appears. Do not reinterpret identical deterministic repeats as independent extra information. No pull, rebuild, source edits, changed worker count or pooling across policy identities during the frozen run. Resume only verified incomplete stages. Raw data/checkpoints/binaries stay ignored; publish compact availability, preflight, sampler, run, map/seat/fault/candidate/repeat coverage and frozen-identity receipts. This pilot measures data coverage/noise/learnability, not BT gain or proof of enough final training data.

## Done when

Reviewed report PR merged: requested-program availability, runtime/A-A preflight, actual sampler masses, frozen inputs/seed/schedule and complete 300-500-map pilot plus candidate/repeat/fault coverage published. Raw whole matches and branches remain on host. No training or strength claim.

## Handoff

Not started.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 16:06Z claude@caspers-macbook: amended depends -> T0035, T0036, T0039, T0040
- 2026-10-12 16:34Z claude-ci@caspers-macbook: amended writes -> bots/p_only/report/decision_collection/, docs/DIGEST.md; needs -> any; title -> Expanded-pool preflight and bounded decision data pilot (run on Colab by Casper's decision, not host evidence)
- 2026-10-12 16:34Z claude-ci@caspers-macbook: claimed (forced)
- 2026-10-12 16:34Z claude-ci@caspers-macbook: note: Casper instructed on 2026-10-12 to run T0037 on Colab instead of the Windows host. Consequences recorded here and in every report: Colab CPU timings and fault rates are not host evidence (AGENTS.md); the data come from the hooked carrier v12_ponly_xxl_dlog compiled and played on Colab, a distinct provenance never pooled with host data. Claimed with --force while T0040 (PR #36, branch-collector fixes, owner claude@caspers-macbook) is in review: stage 1 (availability, preflight, sampler, whole-match collection) runs now; the branch stage waits for #36 to merge. T0032's Colab runtime was deleted 09:40Z, so no shared runtime.
- 2026-10-12 16:35Z claude-ci@caspers-macbook: note: Seed reserved before any game: 20261501 (training-only data, T0037 Colab pilot). Checked unused against the DIGEST ledger (training.decision_opponents.consumed_seeds), the whole repository and existing schedules; it goes into the DIGEST ledger with the report PR. Pilot size predeclared: 336 maps (24 per count, 2-15), one root per map, incumbent plus three logged shortlist candidates, four continuations each; no expansion or early stop from labels.
