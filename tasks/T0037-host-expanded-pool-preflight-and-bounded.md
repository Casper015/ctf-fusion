---
id: T0037
title: "Expanded-pool preflight and bounded decision data pilot (run on Colab by Casper's decision, not host evidence)"
status: open
owner: ""
needs: []
writes: [bots/p_only/report/decision_collection/, docs/DIGEST.md, docs/MARKDOWN_INDEX.md]
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

*2026-10-12 18:08Z codex-t0037-report@caspers-macbook:*

Stage-1 report submitted as PR #37 https://github.com/Casper015/ctf-fusion/pull/37, head 1df0afbd1c7287f071a1fc96aa0e081f0b9b3d33. Source receipts pin 41022c0, engine 54139d4; seed 20261501 consumed; frozen pilot 06b5ccacdafe75ab. 306 complete whole matches, all roots available, two faults retained in s20261501-n09-m00032. Local backup is Player5/workspace/backups/T0037_colab_seed20261501_20261012; archive SHA-256 2d04629757e68d783df5dcb2a093a4a23fb356d8f2a0f41f63ebef07db73e970; 631 file hashes verified. Original Colab /content/t0037 remains intact, binaries still there. Core suite 291 tests OK (43 skipped); task check and diff check pass. This is partial progress, not a completion handoff. Next: review/merge #36 before any branch collection; retain frozen matches/binaries, freeze separate branch collector provenance, collect actual K and four repeats per candidate, report prefix/root/continuation faults and noise. No model training or host strength claim; T0038 remains gated.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 16:06Z claude@caspers-macbook: amended depends -> T0035, T0036, T0039, T0040
- 2026-10-12 16:34Z claude-ci@caspers-macbook: amended writes -> bots/p_only/report/decision_collection/, docs/DIGEST.md; needs -> any; title -> Expanded-pool preflight and bounded decision data pilot (run on Colab by Casper's decision, not host evidence)
- 2026-10-12 16:34Z claude-ci@caspers-macbook: claimed (forced)
- 2026-10-12 16:34Z claude-ci@caspers-macbook: note: Casper instructed on 2026-10-12 to run T0037 on Colab instead of the Windows host. Consequences recorded here and in every report: Colab CPU timings and fault rates are not host evidence (AGENTS.md); the data come from the hooked carrier v12_ponly_xxl_dlog compiled and played on Colab, a distinct provenance never pooled with host data. Claimed with --force while T0040 (PR #36, branch-collector fixes, owner claude@caspers-macbook) is in review: stage 1 (availability, preflight, sampler, whole-match collection) runs now; the branch stage waits for #36 to merge. T0032's Colab runtime was deleted 09:40Z, so no shared runtime.
- 2026-10-12 16:35Z claude-ci@caspers-macbook: note: Seed reserved before any game: 20261501 (training-only data, T0037 Colab pilot). Checked unused against the DIGEST ledger (training.decision_opponents.consumed_seeds), the whole repository and existing schedules; it goes into the DIGEST ledger with the report PR. Pilot size predeclared: 336 maps (24 per count, 2-15), one root per map, incumbent plus three logged shortlist candidates, four continuations each; no expansion or early stop from labels.
- 2026-10-12 17:02Z claude-ci@caspers-macbook: note: Colab preflight (2 vCPU AMD EPYC 7B12, 12 GB; ctf-fusion 41022c0, engine 54139d4; frozen identity ff7872ab, pool 0e2bfcf6, all 52 enabled opponents + designated v12_ponly_xxl_dlog 9d713366 built in 13 min). Feasibility rule declared before probing: zero faults and zero late replies for every seat in every probe (recorded inputs, sampled compositions, 2000/50 ms). Result: counts 2-10 feasible; 11-15 infeasible on this runtime (faults 2-12 per probe; the carrier's refinement cut on 10-46% of turns). Pilot redeclared before any game (Casper chose to run only the feasible counts here): seed 20261501, counts 2-10, 34 maps per count = 306 maps, the same fixtures as the frozen full schedule for maps 0-23. Counts 11-15 remain uncollected and are reported infeasible on Colab. A/A on recorded input: designated log carrier orders identical to production v12_ponly_xxl on 400/400 turns.
- 2026-10-12 18:04Z codex-t0037-report@caspers-macbook: note: Casper explicitly asked Codex to continue checking, back up Colab data, and organize/submit a report PR. Taking over report preparation only; branch collection remains gated on T0040. Local archive hash 2d04629757e68d783df5dcb2a093a4a23fb356d8f2a0f41f63ebef07db73e970 verified against Colab; all 631 file hashes pass, 306 matches and 306 rows retained.
- 2026-10-12 18:04Z codex-t0037-report@caspers-macbook: claimed (forced)
- 2026-10-12 18:06Z codex-t0037-report@caspers-macbook: amended writes -> bots/p_only/report/decision_collection/, docs/DIGEST.md, docs/MARKDOWN_INDEX.md
- 2026-10-12 18:08Z codex-t0037-report@caspers-macbook: handoff -> open: Casper requested report PR submission; PR #37 publishes completed whole-match stage only. Returning task open while T0040/branch stage remains outstanding, rather than claiming full pilot completion.
