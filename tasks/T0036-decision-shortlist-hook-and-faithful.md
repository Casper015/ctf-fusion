---
id: T0036
title: "Decision shortlist hook and faithful branch collector (code only)"
status: active
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

Not started.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 09:18Z claude-ci@caspers-macbook: claimed (forced)
- 2026-10-12 09:18Z claude-ci@caspers-macbook: note: Claimed with --force while T0035 is in review (PR #32): both are held by claude-ci@caspers-macbook at the user's request to code T0035 and T0036; T0036's branch is stacked on T0035's and its PR will target it until #32 merges.
- 2026-10-12 09:24Z claude-ci@caspers-macbook: amended writes -> bots/p_only/bots/c08_ponly.hpp, bots/p_only/bots/c08_ponly_f.hpp, bots/p_only/bots/v12_ponly.hpp, bots/p_only/bots/v12_ponly_f.hpp, bots/p_only/bots/variants/, bots/p_only/include/decision_hook.hpp, bots/p_only/profiles/variants_decision.json, bots/p_only/tests/test_decision_hook.py, bots/p_only/tools/derive/make_c08.py, bots/p_only/tools/derive/make_families.py, bots/p_only/tools/derive/make_v12.py, tests/test_decision_branch_collect.py, training/decision_branch_collect.py
