---
id: T0034
title: "Decision learning: action-gain scorer and reliable evaluation (code only)"
status: active
owner: claude-ci@caspers-macbook
needs: []
writes: [docs/DIGEST.md, docs/MARKDOWN_INDEX.md, docs/RL_PLAN.md, fusion/decision_features.py, tests/test_decision_learning.py, training/decision_dataset.py, training/decision_noise.py, training/evaluate_decisions.py, training/export_mixed.py, training/export_weights.py, training/train_decision_ranker.py]
depends: [T0011, T0030]
land: pr
branch: claude-ci/t0034-decision-learning-action-gain-scorer-and
updated: 2026-10-12
---

## Goal

### Execution prompt: learn better decisions on top of the frozen P/search bot

Work in https://github.com/Casper015/ctf-fusion. Implement the code and evaluation foundation for decision improvement. This is a code-only task: do not launch real training, new matches, continuation batches, deployment timing, or model exports. Do not start subagents. Deliver a reviewable PR and a clear host-run handoff using the repository task workflow.

#### Direction and boundaries

Casper has retired the unstarted supervised-V pilot T0012 and its deployment screen T0014. Preserve their history and all completed experiments. Do not revive their M/4XL V sweep, V distillation, or leaf-value coefficient tuning. Keep deployed V disabled. An action scorer is a separate experimental component; do not disguise it as the existing V or overwrite opponent P artifacts.

P predicts opponents. The new component chooses our own complete actions or, in a later experiment, team tasks. The initial experiment learns relative improvements over the incumbent within a legal shortlist. Its benefit is unproven. Lower loss, more changed decisions, and richer features are not strength claims.

T0032's P3 teacher/student/QAT work continues independently. The code must accept a frozen original P3 or a subsequently admitted distilled P3 through explicit identity/configuration, without requiring either model to finish now. Final data collection and deployment evaluation must bind to the actual selected carrier, features, weights, temperatures, search settings and budget. Do not interfere with the active Colab work.

Read AGENTS.md, CONTRIBUTING.md, tasks/SUMMARY.md, your task, docs/RL_PLAN.md, docs/FEATURE_FAMILIES.md, and docs/P_MIXED_PRECISION.md before implementation. Claim your actual task ID, use its isolated worktree, respect named write lanes, and use the codex/ branch prefix. Never manually edit task files. Amend lanes through task.py before touching another path; wait if another task owns it. Do not change production defaults.

#### Existing components to reuse

- fusion/matchlog.py, fusion/protocol.py and fusion/native.py for recorded inputs, order parsing and native extraction.
- fusion/labels.py for final pairwise utility and canonical labels; retain differences between requested orders and effective/canonical actions.
- training/branch_collect.py, training/branch_dataset.py and training/branch_portfolio.py for common-prefix branches and complete-order candidates.
- training/evaluate_value.py for map-clustered comparisons and incumbent-relative regret. Its true-successor ranking is not a deployment-condition gate by itself.
- fusion/torch_backend.py and the existing cache/training utilities. Prefer a small scorer; no large teacher is required for this task.
- bots/p_only/report/v_data/ for T0010 provenance and the strict host re-audit implementation in REPRODUCE.md.
- reports/models/p_features/ (PR #21), bots/p_only/report/search_portfolio/ (PR #25), bots/p_only/report/near_far/ (PR #26), and reports/models/p_ood/ (PR #27 or its published head).

T0010 contains p2 continuations: 13,702 retained roots, four candidates, one repeat each. It is suitable for audited exploratory work, not a final P3 decision-policy gate. Check the current DATA_AUDIT.json, not the task's done status. Refuse authoritative rank-data ingestion while strict source alignment is pending; full-manifest versus selected-identity verification scopes must remain explicit. Never reconstruct a freeze-time manifest baseline from current bytes. Never reuse consumed measurement seeds.

#### Implement the minimum useful experiment

1. Add a versioned root/candidate dataset contract and builder. Store match/map/count/seat/turn, split, carrier identity, incumbent index, canonical complete-order identity, candidate source, raw per-repeat final score vectors, derived outcomes and repeat counts. Preserve every candidate/drop reason and uncertainty; never silently manufacture missing labels or probabilities.

2. Reconstruct our root features from the recorded prefix with the exact runtime history contract. Do not obtain a root record by pretending it was the first turn of a one-turn match. Bind per-candidate movement, pickup/wait/drop/attack choices and target features to the root. Audit legal/support masks, duplicate effective actions, out-of-cap attack targets and aliasing of distinct candidates. A target outside the representation is unavailable, not a zero-loss success or a probability silently clipped to epsilon. Score inputs may not contain future outcomes, raw future opponent replies, or true realized successors unknown at decision time.

3. Add noise/coverage diagnostics. Measure within-root, within-action repeat variation separately from between-action variation. Original-game versus replay differences are a separate memory/replay diagnostic. One repeat cannot establish repeat variance. Report effective map counts, per-count coverage, feature collisions, changeable roots and label availability. Do not adopt r_AA >= 0.8*r_cross or a 50% feature-collision rate as automatic kill rules. Do not infer continuation noise solely from the spread of effects across different roots.

4. Implement a small direct complete-action scorer Q_theta(history, candidate). Train incumbent-relative gains or within-root preferences using repeated-outcome uncertainty when available. The exploratory target is A^pi0(s,a) = E[U_T | s,a, then pi0] - E[U_T | s,a0, then pi0]. Use a root-shared/state-action representation that can distinguish complete team orders, with deterministic candidate ordering and masks. Keep the opponent P frozen. Add an explicit model kind such as own_action_scorer, and prevent it from being exported/loaded as opponent P or legacy V. Start with simple baselines before increasing capacity. DPO may remain a separately gated follow-up, not an obligatory prerequisite; do not implement PPO/CQL/IQL or six-round iteration here.

5. Implement evaluation of the exact intended decision rule. Include the incumbent, a simple fitted ranker and the learned scorer. If deployment uses S_normalized + lambda*Q, choose lambda and the normalization on validation and freeze both before test. Evaluate that same rule, not pure argmax Q. All roots, including unchanged and uncertain roots, belong in the primary incumbent-relative utility denominator. Weight team counts equally and cluster by independent map. Preserve train/val/test isolation across every root of a map; lock selection/configuration identities before reading test outcomes.

6. Provide an optional diagnostic oracle using real successor states, clearly labeled unavailable at deployment. A V/simulator comparison must choose using P-predicted opponent actions at deployment conditions. Do not claim a true-successor test is unbiased evidence for a predictor that cannot observe those actions. If estimating candidate headroom, use one independent repeat subset to select candidates and another to evaluate them; do not maximize and evaluate on the same noisy outcomes. Report uncertainty instead of fabricating headroom when repeats are insufficient.

7. Specify the subsequent carrier hook and host pilot without running either. The shortlist must always include the legal incumbent, deduplicate complete orders, and expose actual K, source/score and decision coverage. With the experiment off or lambda=0, bypass all extra work and preserve the incumbent path. Reserve time for feature encoding, scoring, search, protocol parsing and output within the full 50 ms limit; a nominal extra 3 ms is not free. Reuse immutable P features/views, candidate buffers and scorer caches where the complete history/state/model identity permits. No cached result may leak across branches or model versions.

#### Macro decisions and later RL

Document a later team-task experiment over the carrier's existing task assignment: delivery/escort, interception/defense, flag selection and disengagement. The actions must be executable complete team plans with explicit replanning/termination conditions, not informal tactical labels. Micro legality, damage allocation and movement stay with the existing mathematical layer. Measure whether the incumbent's macro candidate set actually contains better plans before training another model.

Expert iteration requires either a stronger offline expert or a demonstrated ability of a student to save useful online search work. Compare imitation of the original search against imitation of a larger-budget expert. Fresh on-policy data is required for validating the updated deployment policy, while correctly labeled old-policy data may be used for exploratory pretraining/replay. A failed DPO arm does not automatically rule out stronger-expert distillation.

Before multi-step offline RL, require complete transitions: root/history, actual legal shortlist, selected action, actual behavior probability, next state/history, rewards and termination. For exploration among K distinct candidates, use the actual K and handle K=1. Keep the final pairwise-utility objective. If using r_t=U_{t+1}-U_t, explicitly require an undiscounted complete episode so the sum equals U_T-U_0. Do not add arbitrary capture/kill/distance bonuses to the primary objective. Opponents can have adaptive internal memory even when their binaries are frozen.

#### Tests, delivery and host handoff

Use small fixtures/recorded inputs, not formal match batches. Tests must reject incorrect source linkage, missing history, outcome leakage, wrong on-policy metadata, invalid support, duplicate candidates, split contamination and model-kind misuse. Verify repeated-outcome arithmetic, per-root/count weights, map clustering, deterministic tie behavior and frozen selection identities. Run the required repository checks appropriate to touched files.

Revise docs/RL_PLAN.md to reflect the retired V tasks and this conditional route. Distinguish evidence of benefit, harm, insufficient precision and a predeclared practically negligible effect; a CI containing zero is not proof that learning is impossible. Plan power using train/val map-level variance rather than assumed independent roots or test-set tuning. Predeclare primary model comparisons or account for multiple comparisons.

The host handoff should propose a bounded pilot after the deployment carrier is frozen: approximately 300-500 independent maps, one representative root per map, incumbent plus three legal shortlist candidates, and initially four continuations each. Size repetitions and any expansion from actual measured noise and compute cost; no timing/BT forecasts are evidence. Use independent repeat subsets for headroom selection/evaluation, and separate map splits for learning and testing. Only an equal-full-turn-time fresh-map screen can establish deployed improvement. Keep an A/A control, all faults/timeouts, actual valid denominators and unchanged production control.

Finish with a PR containing code, meaningful tests, the revised plan and exact proposed host commands. State plainly: no real training, host measurements, new games, model exports or strength improvement occurred in this code task. Leave the board with an accurate handoff through task.py. Do not automatically create follow-up training tasks, consume seeds, deploy models or merge the PR.


## Done when

A reviewed code-only PR delivers the versioned history/candidate dataset contract, a small own-action scorer, leakage/support/noise diagnostics, deployment-rule-matched map-cluster evaluation, model-kind separation and meaningful tests. docs/RL_PLAN.md records the retirement of T0012/T0014 and the conditional macro/expert-iteration path. Include exact proposed host commands and a bounded pilot specification, but run no real training, measurements, matches, continuation batches or model exports. Benefit remains unproven; the next agent claims this task before coding.

## Handoff

Not started.

## Log

- 2026-10-12 05:58Z codex-rl-plan@caspers-macbook: created
- 2026-10-12 06:02Z claude-ci@caspers-macbook: claimed
- 2026-10-12 06:16Z claude-ci@caspers-macbook: amended writes -> docs/DIGEST.md, docs/RL_PLAN.md, fusion/decision_features.py, tests/test_decision_learning.py, training/decision_dataset.py, training/decision_noise.py, training/evaluate_decisions.py, training/export_mixed.py, training/export_weights.py, training/train_decision_ranker.py
- 2026-10-12 06:18Z claude-ci@caspers-macbook: amended writes -> docs/DIGEST.md, docs/MARKDOWN_INDEX.md, docs/RL_PLAN.md, fusion/decision_features.py, tests/test_decision_learning.py, training/decision_dataset.py, training/decision_noise.py, training/evaluate_decisions.py, training/export_mixed.py, training/export_weights.py, training/train_decision_ranker.py
- 2026-10-12 06:21Z claude-ci@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/28
