---
id: T0036
title: "Decision shortlist hook and faithful branch collector (code only)"
status: review
owner: claude-ci@caspers-macbook
needs: []
writes: [bots/p_only/bots/c08_ponly.hpp, bots/p_only/bots/c08_ponly_f.hpp, bots/p_only/bots/v12_ponly.hpp, bots/p_only/bots/v12_ponly_f.hpp, bots/p_only/bots/variants/, bots/p_only/include/decision_hook.hpp, bots/p_only/profiles/variants_decision.json, bots/p_only/tests/test_decision_hook.py, bots/p_only/tools/derive/make_c08.py, bots/p_only/tools/derive/make_families.py, bots/p_only/tools/derive/make_v12.py, tests/test_decision_branch_collect.py, training/decision_branch_collect.py]
depends: [T0035, T0034]
land: pr
branch: claude-ci/t0036-decision-shortlist-hook-and-faithful
updated: 2026-10-12
---

## Goal

Read AGENTS.md and CONTRIBUTING.md, claim this task and use its isolated worktree. Respect active/review write lanes and machine locks. All repository prose is English. Do not spawn subagents. Preserve completed experiments and consumed seeds. Do not reopen T0012/T0014, change production defaults, or interfere with T0032. Implement the shortlist logging and branch-collection interfaces needed by the decision-learning pilot. Code only: no formal collection, training or host timing. Use the registry/collector supplied by T0035 and the own-action contracts merged through T0034/PR #28.

Instrument the actual C08/v12 incumbent planners behind off-by-default generator-controlled flags. Log actual complete shortlisted orders, their true carrier scores, incumbent identity, sources, K and root digest using ctf-fusion-shortlist-scores-1. Include the incumbent, deduplicate effective orders and allow K=1. Do not replace missing scores with fabricated values. Runtime feature encoding/canonicalization must match fusion/decision_features.py; original P2 is permitted despite richer scorer inputs.

With the hook disabled or lambda=0, preserve original replies and skip extra scoring work. Distinguish log-only collection mode from deployed scoring. If a scoring interface is included, use the exact zero-weight/incumbent-anchored rule from training/evaluate_decisions.py; export/load an own_action_scorer through its own route, never opponent P or legacy V. Do not require an actual trained model to test logging. Regenerate headers through their source generators and preserve root/branch isolation and full-turn deadlines.

Build a branch collector over logged actual shortlists (incumbent plus up to three alternatives), using frozen commands/identities from T0035. Accept and retain four per-candidate repeat outcomes initially. The full recorded prefix is required. Every branch reproduces the same engine root; root-turn opponent actions are common across candidates, with opponents live/reactive afterward. Restore/commit the forced order consistently for our controlled bot's internal state where supported; otherwise label the forced-action/memory approximation explicitly. Log original/replayed replies, prefix mismatches, RNG/repeat keys, bot-state mechanism, faults and root/successor blocks. Do not promise perfect black-box memory restoration or zero noise. Controlled deterministic/fixed-work diagnostics must be separately tagged from the actual 50ms continuation policy.

Serialize raw per-repeat final score vectors and derived utility, exact candidate orders and true carrier_scores for decision_dataset ingestion. Resume only under unchanged source/binary/engine/runner/config/limits identities. Never hide missing roots, legal shortages, timeouts or unsupported candidates. Add recorded-input/toy tests for zero/off behavior, logging/order-score alignment, root invariance, repeat provenance, score derivation and collector source mismatch. Define the C++/Python feature parity test and host budget command without running formal host measurements here.

## Done when

Reviewed code PR merged: actual complete shortlist/score logging, disabled/zero-weight equivalence, root/history/source contracts and repeated-branch serialization tested; CPU budget/parity checks specified for host. No formal host collection or trained-model deployment.

## Handoff

*2026-10-12 09:38Z claude-ci@caspers-macbook:*

PR https://github.com/Casper015/ctf-fusion/pull/33, stacked on PR #32 (T0035): its base is T0035's branch; retarget it to main after #32 merges. Claimed with --force while T0035 was in review (both held by claude-ci@caspers-macbook). Code only: no collection, training, deployment or host timing; no seed consumed.

Done (d186034 on 070a517): decision_hook.hpp (PONLY_DECISION 0 off / 1 log-only / 2 refused), make_families.py edits and regenerated v12_ponly_f.hpp/c08_ponly_f.hpp (production v12_ponly.hpp/c08_ponly.hpp untouched), variants v12_ponly_xxl_dlog/c08_ponly_xxl_dlog with bots/p_only/profiles/variants_decision.json (role decision_log), training/decision_branch_collect.py, tests (P-only test_decision_hook 4, core test_decision_branch_collect 11; P-only suite 50 OK/3 skipped, core 291 OK/43 skipped). Locally on recorded inputs: Python root digest equals the hook's on 1600 turns; off and log-only orders identical.

Needed before T0037 (one small config change once #32 merges, in T0035's files): profiles/decision_opponents.json designated.variant -> v12_ponly_xxl_dlog; profiles/decision_variants.json catalogs += bots/p_only/profiles/variants_decision.json and exclude_roles += decision_log (so the log carriers are never opponents). Data from this hooked family build is a new carrier version (Refresh rule); do not pool it with v12_ponly_xxl data.

Host (not run here):
  1. Budget: python bots/p_only/tools/build.py --variants bots/p_only/profiles/variants_decision.json --stats, then a paired equal-time timing screen (bots/p_only/tools/screen_runner.py) of v12_ponly_xxl vs v12_ponly_xxl_dlog with an A/A copy on a fresh unused seed: full-reply first/p50/p99/max, late replies and faults at 1/6/15 concurrent copies, all inside the official 50 ms; the log must add no lateness.
  2. Collection: T0035's decision_collect list/schedule/prepare/collect with the hooked designated carrier, then
     python tools/repo/task.py lock cpu --task T0037 -- python -m training.decision_branch_collect --frozen build/decision/run1/FROZEN.json --matches data/decision/run1/matches --out data/decision/run1/branches --repeats 4 --alternatives 3 --workers 6 --ai-test $AI_TEST_ROOT
  3. Dataset: python -m training.decision_dataset --branches data/decision/run1/branches --matches data/decision/run1/matches --carrier-config <derived from the run's receipts> --out data/decision/run1/dataset (authoritative only with a strict audit receipt from source-digests).

Specified, not implemented: C++/Python feature-parity test for a future scoring route (encode the root and each shortlisted order in C++ with the extractor's encoders and compare with fusion.decision_features.root_features/candidate_features on recorded roots, bit for bit after scaling). Not checked: no 15-team hook replay in tests (v12's refine deadline makes it machine-dependent); C08 'checked' turns where the lookahead replaced the search choice are reported, not scored (11 of 400 on the n04 fixture).

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 09:18Z claude-ci@caspers-macbook: claimed (forced)
- 2026-10-12 09:18Z claude-ci@caspers-macbook: note: Claimed with --force while T0035 is in review (PR #32): both are held by claude-ci@caspers-macbook at the user's request to code T0035 and T0036; T0036's branch is stacked on T0035's and its PR will target it until #32 merges.
- 2026-10-12 09:24Z claude-ci@caspers-macbook: amended writes -> bots/p_only/bots/c08_ponly.hpp, bots/p_only/bots/c08_ponly_f.hpp, bots/p_only/bots/v12_ponly.hpp, bots/p_only/bots/v12_ponly_f.hpp, bots/p_only/bots/variants/, bots/p_only/include/decision_hook.hpp, bots/p_only/profiles/variants_decision.json, bots/p_only/tests/test_decision_hook.py, bots/p_only/tools/derive/make_c08.py, bots/p_only/tools/derive/make_families.py, bots/p_only/tools/derive/make_v12.py, tests/test_decision_branch_collect.py, training/decision_branch_collect.py
- 2026-10-12 09:38Z claude-ci@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/33 (stacked on #32)
- 2026-10-12 09:38Z claude-ci@caspers-macbook: handoff -> review
