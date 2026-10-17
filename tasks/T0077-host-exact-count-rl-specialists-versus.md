---
id: T0077
title: "Host: exact-count RL specialists versus general control and common initialization"
status: open
owner: ""
needs: [host, ai-test]
writes: [reports/host/agent_v2_rl_acceptance/, profiles/agent_v2_rl_acceptance.json, docs/AGENT_V2_RL_ACCEPTANCE_SEEDS.md]
depends: [T0068, T0075]
land: direct
branch: ""
updated: 2026-10-17
---

## Goal

Evaluate T0068 as its spec requires. Arms, all actor-direct agent_v2 with the CSP P bank: S = agent_v2_rl_csp (14 exact-count specialists), G = the compute-matched general control, I = the common initialization actor (G and I from T0075), plus an A/A copy of S. 96 fixtures per count 2-15 = 1,344 paired fixtures, 5,376 formal games (about 90% power for +2 pp with two Bonferroni contrasts under T0073's variance). Proposed seeds: 20262100 preflight, 20262101 screen. Primary S-G and supporting G-I form one two-contrast Bonferroni family; S-I and per-count rows are descriptive. Report the actual training exposure: the general arm got 222-279 episodes per count (reports/models/count_specialists_rl/REPORT.md), not exactly 250. No retraining, checkpoint selection or promotion to default. Shared evaluation contract (N2, N3, N5): fresh paired fixtures share map, flag randomness, seat and opponent schedule across arms; counts 2-15 weighted equally; held-out roster Fable/MiMo/Kimi/GLM/Player4 (profiles/meow_rl_screen.json); exactly eight workers under task.py lock cpu, no GPU training running, 2,000 ms first / 50 ms later replies. Preflight every arm on 16 fixtures at each count 2/5/8/15 (zero all-seat faults, complete telemetry, first reply < 1,800 ms, steady reply < 48 ms); a failure is recorded and blocks the screen, never fewer workers. Freeze binary/model/engine/runner hashes, profile and schedule before play; finish the fixed budget without outcome-based stopping; stop only on integrity or safety failure. Map-cluster 95% intervals, margin zero, multiplicity via the runner's multiplicity field (bots/p_only/tools/run_screen.py). 2-team fixtures were saturated vs this roster in T0057/T0073 (zero difference, zero spread): keep the 14-count primary, and also report a predeclared descriptive equal-weight 3-15 estimate and the 2-team band separately. Seed bases below were free on 2026-10-17: recheck every ledger and derived-seed collision, then reserve before play.

## Done when

Seeds reserved before play; a passed preflight receipt (or a blocked handoff with the failure); complete frozen results under reports/host/agent_v2_rl_acceptance/ with A/A score-vector agreement, identities, Bonferroni-adjusted intervals, the 3-15 and 2-team descriptive rows, fault and timing totals, and a report that separates specialization (S-G) from the PPO gain (G-I).

## Handoff

Not started.

## Log

- 2026-10-17 04:07Z claude-review@caspers-macbook: created
