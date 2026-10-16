---
id: T0068
title: "Train exact-count PPO policy and value specialists"
status: active
owner: antigravity@desktop-g60ojud
needs: [host, cuda, ai-test]
writes: [reports/host/count_specialists_rl/, reports/models/count_specialists_rl/, profiles/count_specialists_rl.json, docs/COUNT_SPECIALISTS_RL_SEEDS.md, inference/count_specialists_rl/]
depends: [T0065, T0067]
land: pr
branch: antigravity/t0068-train-exact-count-ppo-policy-and-value
updated: 2026-10-16
---

## Goal

Use the merged modular T0065 carrier and the frozen refreshed P bank to train distinct own-policy pi/V specialists for each exact count 2-15. Mathematical resolver and scoring stay fixed. No CNN or simultaneous model-size sweep. Inspect T0064 outcomes and the new carrier's readiness before freezing configuration; do not silently transfer old code-digest identities.

Create fresh on-policy full games for each nXX actor; static supervised P logs and old root continuations are not PPO rollouts. Warm-start from a common general actor only when compatible with the NEW carrier. If no compatible actor exists, first prepare a separately budgeted general initialization/training task; do not squeeze new features/actions into legacy schemas to load old weights. Use the same declared initialization protocol across specialist and general-control arms. Warm-start V only with matching features/reward units and label it initialization; otherwise initialize fresh. Keep actor/critic architecture, PPO settings, opponents, precision and resolver identical across specialized and general-control arms. The specialized arm has 14 policies, each updated only on its own count; record joint action log probabilities under the exact deployed P/resolver/history path.

Bounded first experiment: 100,000 learner decisions per count in the specialized arm (1.4M aggregate), compared against one general policy given the same 100,000 fresh decisions per count (1.4M aggregate). Total initial training budget is 2.8M learner decisions; first 20,000 decisions in each nXX specialized run are its included correctness smoke, not extra budget. Use separate ledgered game seeds, matched roster/curriculum strata and optimizer seed 101; the general arm's data are newly collected under its own policy, not reused specialized trajectories. Arrange complete episodes and record any unavoidable batch rounding; match actual total decisions between arms, without changing the result-driven budget. Interleave general-arm counts; do not train sequentially on one count and forget the others. No unbounded expansion or best-checkpoint selection on test.

Training pools exclude final evaluation lineages. Reserve independent train/calibration/evaluation seeds and maps per count. Freeze final checkpoints, and align calibration corpus per nXX across its two compared policies. Eight-worker CPU rollout/preflight and official 2s/50ms opponent limits, plus GPU lease for learning; no concurrent independent host measurements. Preserve full state/RNG/optimizer/checkpoint receipts, all-seat faults and recovery semantics.

Before any formal deployment batch, create a dependent evaluation task with fixed new seeds, schedule, primary specialist-bank minus general-control contrast, A/A, exactly eight workers, official reply limits, equal weight for each of the 14 counts, map-cluster intervals and multiplicity for per-count claims. Training utility and single-seed results cannot prove general architecture effects. This task does not authorize an unspecified match batch. Larger training budgets, three-seed RL replication and different model widths by count are future tasks, informed by this pilot.

## Done when

Fourteen specialized pi/V checkpoints plus the compute-matched general control, fresh causal per-count PPO receipts, verified native inference parity and backups, independent data/calibration identities, and ready registry routing delivered in a PR. Genuine training only after upstream code/model gates; no claim that this task's registration launched a GPU process. Exact-eight-worker host feasibility established before creating the frozen evaluation task. Negative or inconclusive outcomes retained; no automatic default promotion.

## Handoff

*2026-10-16 00:12Z codex-count-plan@caspers-macbook:*

User-authorized future host work. Read Goal and Done when in full, inspect current upstream tasks, then claim with the required capabilities. These tasks are newly registered; no collection/training was launched by the planning agent. Keep ongoing T0064 host work and T0065 carrier implementation isolated. Specialized data and policies use exact counts 2-15, not band labels.

## Log

- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: created
- 2026-10-16 00:12Z codex-count-plan@caspers-macbook: handoff prepared
- 2026-10-16 00:25Z codex-architecture@caspers-macbook: updated at Casper's explicit request; redesign permits changed behavior; preserved task owner/status/branch; gated fresh collection on T0065.
- 2026-10-16 08:12Z claude-review@caspers-macbook: note: Casper's decision (2026-10-16): Player2 is held out. profiles/meow_rl_training_pool_v2.json still lists Player2_v005 (family player5); any PPO training pool here must exclude it explicitly.
- 2026-10-16 08:15Z claude-review@caspers-macbook: note: CORRECTION: ignore my note above. Player2_v005 in pool v2 is correct (Player5 lineage, docs/MEOWMIND_RL_POOL_SPLIT.md); no extra exclusion is needed. The held-out Player2 is the original ctf-stress-pack bot. https://github.com/Casper015/ctf-fusion/pull/61#issuecomment-6211153284
- 2026-10-16 20:36Z antigravity@desktop-g60ojud: claimed
