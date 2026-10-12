---
id: T0035
title: "Expanded decision opponent registry and collector (code only)"
status: done
owner: claude-ci@caspers-macbook
needs: []
writes: [profiles/decision_opponents.json, profiles/decision_variants.json, training/decision_opponents.py, training/decision_collect.py, tests/test_decision_opponents.py]
depends: []
land: pr
branch: claude-ci/t0035-expanded-decision-opponent-registry-and
updated: 2026-10-12
---

## Goal

Read AGENTS.md and CONTRIBUTING.md, claim this task and use its isolated worktree. Respect active/review write lanes and machine locks. All repository prose is English. Do not spawn subagents. Preserve completed experiments and consumed seeds. Do not reopen T0012/T0014, change production defaults, or interfere with T0032. Implement an executable expanded opponent registry and designated-carrier whole-match collector. Code only; do not launch real games, training or timing runs in this task.

Roster: retain the twelve T0010 references from bots/p_only/report/v_data/schedule.json (Astra, Sol, Fable, Sonnet, Opus, DeepSeek, Qwen, MiMo, Kimi, GLM, Player5 and Baseline). Add these six fingerprint-screened representatives by stable source ID/hash from roster/report/training_selection_20261009/candidate_selection.csv: ctf-arena_Player5_v12_rollout_volley; cand-012; C04_G_PLAYOUT_4ply; bot_2 = New_AI_Test/Player5/workspace/submission/bot.cpp; ctf-arena_Player5_fortress; ctf-arena_Player5_sparring_raider.

Also register ALL P-only/P+V entries from bots/p_only/profiles/variants.json, variants_families.json and variants_search.json. Merge included catalogs without duplicate names. There are currently 39 neural entries: 27 P-only and 7 P+V have model registries; 5 V-pilot A-E wrappers lack weights and remain explicitly disabled with zero sampling. S uses inference/models, not inference/models_s. Future T0032 student/QAT exports need separate admission; the teacher is offline only. Existing V-on opponents are allowed; our own continuation carrier remains V off.

Target opponent-seat composition: old references 60%, six additions 20%, neural variants 20%. Balance neural C08/v12 groups before versions; aliases/identical binaries share exposure. Reconcile source lineages using the roster inventory, not architecture names. Keep related-family exposure bounded, report cap feasibility/relaxation, and simulate actual per-count masses: without-replacement sampling will not exactly preserve requested targets. Exclude both reserved source lineages AI_Arena_CTF_Player02_bot and ctf-arena_Player4_v2 plus aliases. Do not automatically import the entire old draft open pool.

Use a frozen designated incumbent v12_ponly_xxl/P2, V off, unless a later model receives separate deployment admission. Every fixture must contain exactly one @candidate seat with its binary identity and distinguish opponent label rows from designated own-action rows. Reuse existing runner/matchlog and P-only bundling/build facilities; raw wrapper paths must not be compiled without their transitive includes/model identities. Resolve external AI_Test paths or exact hash-matching vendored aliases, recording substitutions; do not silently substitute different sources.

Provide CLI stages to list availability, prepare/build/freeze, dry-run/simulate the schedule and collect complete raw trajectories. Pin all sources, dependencies, weights, calibration, compiler/flags, binaries, engine, runner, pool and schedule before collection. Missing models, runtime quarantine and alias merges stay inspectable; an exclusion creates a new effective pool identity. Resume must validate identities and completeness, never skip merely by filename or rebuild while measurement is active. Use atomic files and preserve partial/error records. The collector must retain official 2000/50ms limits, all seats/replies/faults and sampling/group/lineage metadata. Include required tests for exclusions, missing weights, catalog inclusion, neural build resolution, deterministic fixtures, designated seats, alias budgets and stale-resume rejection.

## Done when

Reviewed code PR merged: all requested candidates inventoried, disabled/alias states explicit, safe designated-carrier collection and deterministic sampler/resume contracts tested; exact host commands in Handoff. No real collection or training.

## Handoff

*2026-10-12 16:00Z claude-ci@caspers-macbook:*

Merged to main as 54f50f2 (PR #32, including the designated-build fix 070a517). Code only: registry, schedules and collector; no games or seeds used. Host commands in the Handoff.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 09:05Z claude-ci@caspers-macbook: claimed
- 2026-10-12 09:17Z claude-ci@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/32
- 2026-10-12 09:17Z claude-ci@caspers-macbook: handoff -> review
- 2026-10-12 09:38Z claude-ci@caspers-macbook: note: Fix pushed to PR #32 (070a517): the designated carrier now builds into its own directory, so a same-named opponent build (without PONLY_STATS) cannot overwrite its bundle source; regression test added (21 tests). Found while stacking T0036.
- 2026-10-12 16:00Z claude-ci@caspers-macbook: done
- 2026-10-12 16:09Z codex-review@caspers-macbook: note: Code deliverable accepted: https://github.com/Casper015/ctf-fusion/pull/32#pullrequestreview-5493228889. Verified merged and focused registry/whole-match tests pass; no host build/collection/timing here. Follow-ups T0039/T0040 must pass before T0037.
