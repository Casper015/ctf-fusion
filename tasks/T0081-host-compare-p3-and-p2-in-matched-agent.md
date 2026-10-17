---
id: T0081
title: "Host compare P3 and P2 in matched agent_v2 search"
status: open
owner: ""
needs: [host, ai-test]
writes: [reports/host/agent_v2_p3_search/, profiles/agent_v2_p3_search.json, docs/AGENT_V2_P3_SEARCH_SEEDS.md, tools/screens/p3_search_host/]
depends: [T0078, T0079, T0080]
land: direct
branch: ""
updated: 2026-10-17
---

## Goal

Main comparison clarified by Casper: use the GENERAL P2 XXL (inference/models_pf_p2xxl_vm, pf_p2_xxl_s1) versus GENERAL P3 XXL (inference/models_pf_p3xxl_vm, pf_p3_xxl_s1) from the same T0011 feature-family study. Both serve one general policy across counts 2-15. Do not use the count-routed P2 bank as the main control; keep T0079 only as an external deployment reference. Pin both manifests, headers, feature-family and calibration identities. Own actor, FP32 critic, resolver, breadth, scenario keys, tau and deadline remain fixed. This clarification supersedes any older routing description below.

Execute the P3 prediction experiment in bots/agent_v2/p3_search/README.md after its code PR and T0079 finish. No retraining or default promotion. Four arms use the same aligned T0059 actor, resolver and FP32 critic: P2 direct, P3 direct, P2 gated 8x4 search and P3 gated 8x4 search, tau 0.002. Use the existing P3 XXL bank (T0011 general model) through native P3 features/action/target heads at root and branches. This is a prediction-package comparison including feature/weight/routing differences, not proof of a single feature's causal effect.

First run matched fixed-work diagnostics with existing public causal prefixes: identical input observations, actor, candidate list, random keys, tree breadth and critic. P2/P3 may produce different modeled orders and successors. Do not interpret a higher V prediction under its own model as real improvement. Preserve and bind code/model/tree receipts.

Register fresh seeds before all new games. Exactly eight game workers, CPU lease, no GPU training, first reply <=2s and later full replies <=50ms. Preflight 16 paired fixtures at each count 2/5/8/15 for all four variants, 256 games; bind binary/bundle/model/engine/runner/settings hashes, all-seat faults, counter coverage, search_used/overrides/fallbacks/leaves and latency per count. Search completion must be nonzero in each count and comparable work must be reported; otherwise stop as compute-limited rather than spending a full screen on fallback copies. Do not silently narrow the P3 tree or adjust tau/deadline. If a backend fails at eight workers, preserve failures and register new settings/seed before a new preflight.

If the gate passes, freeze 56 maps per count 2-15, 784 independent paired fixtures, four arms plus P3-search A/A = 3920 games. Held-out roster audited against both P training corpora: report overlap explicitly; use approved available lineages, not an unsupported OOD claim. Main comparisons P3search-P2search, P3direct-P2direct, P3search-P3direct form one family (Bonferroni 3). Equal count weighting, map-cluster intervals, fixed budgets, interleaved arm order, full failure/missingness accounting; no favorable early stopping, no added games after inspection. Report prediction cost and completed-work differences along with utility. Use fresh schedules rather than modifying T0079. Publish results or a documented blocked gate, with verified backups.

## Done when

Pinned variants and identity-bound fixed-work diagnostics delivered; eight-worker preflight passed or documented as blocked. If passed, all 3920 scheduled games analyzed under the three-comparison rule with A/A and full telemetry. Report helps/hurts/not shown without inventing a P3 improvement or conflating higher prediction accuracy with stronger decisions. No production defaults changed.

## Handoff

Not started.

## Log

- 2026-10-17 07:50Z codex-p3search@caspers-macbook: created
- 2026-10-17 07:56Z codex-p3search@caspers-macbook: amended goal
- 2026-10-17 08:58Z claude-review@caspers-macbook: amended depends -> T0078, T0079, T0080
- 2026-10-17 08:58Z claude-review@caspers-macbook: note: Casper (2026-10-17): run T0078 (original Meowmind search on/off) before this task. T0079 found agent_v2's gated search loses to direct play (-3.10 pp; -7.90 pp at 3-5 teams). Recommendation: run this P3-vs-P2 search screen only if T0078 shows search can help in this game (ON-OFF > 0); otherwise reconsider.
