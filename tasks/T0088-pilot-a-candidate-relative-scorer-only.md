---
id: T0088
title: "Pilot a candidate-relative scorer only after measured branch headroom"
status: open
owner: ""
needs: [host, cuda, ai-test]
writes: [training/agent_v2_ranker/, inference/agent_v2_ranker/, bots/agent_v2/ranker/, tools/search_diagnostics/ranker/, reports/models/agent_v2_ranker/, profiles/agent_v2_ranker_pilot.json]
depends: [T0082, T0086, T0092]
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

Conditional small experiment, not a presumed cure. Start only if the branch diagnostic decision supports measurable candidate opportunity and weak existing ranking with usable labels; otherwise deliver a defer decision, no training. Reuse audited decision-dataset, scorer export and noise-estimation mechanisms from T0038/T0042/T0044/T0050, but never load old v12 labels or an old scorer as an agent_v2 checkpoint. Do not invade T0051's active lanes. Keep P, actor, resolver, candidate support and continuation policy frozen; select one based on validation, not a post-hoc winner from final games.

First implement a baseline-relative feature/label contract with no future or hidden-state inputs: current public context + complete candidate order + baseline order + optionally reproducible modeled successor features. f(s,a)-f(s,a0) must be exactly zero at a=a0. Supervise signed terminal utility differences under the frozen continuation policy, with bounded noise-aware weighting; ambiguous pairwise labels must not become hard certain preferences. Add a separate public-feature alias audit before widening any network. Report ties, calibration, within-root rank accuracy and selected-action gain on independent continuation repeats. Do not optimize solely for a rank correlation.

Use the upstream pilot's fixed map splits; normalization and fitting train only, architecture/threshold selection validation only, diagnostic test once. This remains a small-data feasibility study. Compare direct baseline (zero gain), the frozen existing V ranking, a regularized linear scorer and one small 32-hidden-unit MLP, seeds 101/102/103; at most six fits, ten epochs per fit, one predeclared learning rate/regularization setting per model, validation early stopping. Freeze the numeric settings/profile before fitting. No automatic extra rows, hyperparameter sweep, larger model or repeated test selection. If insufficient data/uncertainty blocks a conclusion, deliver a costed collection proposal and stop.

Export only a validation-selected candidate after Python/native parity and INT16 or FP32 CPU feasibility checks (choice frozen before diagnostic test); pin feature/model/calibration hashes. Zero-scorer and disabled integration reproduce the direct baseline; full orders always remain legal. Initially compare the scorer as an isolated root-candidate selector without changing the existing search scheduling; any use inside staged search is a separate combination arm. Do not interpret supervised branch scoring as PPO/RL training, a new optimal V, or a guaranteed Meowmind-sized gain. Deployment changes the future policy and hence label distribution; any later policy-iteration recollection is a separate task, not free reuse as on-policy data.

Shared execution contract: this continues an architecture redesign, not a legacy-behavior refactor. Keep official rules exact; do not require whole-bot parity with Meowmind. Preserve agent_v2_p2xxl_direct as production default. P (opponent prediction), pi (our actor), and V/advantage scorer are distinct components. Follow AGENTS.md, claim with your own identity, respect active lanes (especially T0082/PR69 and T0051), and use isolated worktrees. All new timing, collection and strength evidence comes from desktop-g60ojud with exactly eight game workers under the CPU lease, GPU training stopped, 2000/50 ms complete replies. GPU training uses the GPU lease and is scheduled separately. Pre-register fresh map/flag/actor/scenario seeds and check derived-seed collisions; never replay consumed seeds as new evidence. Offline reanalysis of existing immutable artifacts is allowed. Freeze source, engine, model, runner, binary and profile hashes. Missing evidence is unavailable, never zero or a successful check. Do not extend budgets after inspecting outcomes. No default promotion in these tasks; no new giant P/V, CNN, count specialists, or full learned attack policy. All repository prose is English.


## Done when

A reviewed code/model PR or a documented defer/no-signal decision, with immutable six-fit-or-fewer receipts, map-disjoint validation/test, baseline comparisons, actual export parity and a precise inference contract. Only a supported validation-selected candidate is eligible for a host screen. No default changed and no large training run launched.

## Handoff

Not started.

## Log

- 2026-10-18 03:36Z codex-search-plan@caspers-macbook: created
- 2026-10-18 17:22Z claude-fervent@cloud: amended depends -> T0082, T0086, T0092
