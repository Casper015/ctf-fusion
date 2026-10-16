---
id: T0073
title: "Host: screen agent_v2 with per-count P bank vs XXL P"
status: done
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [reports/host/agent_v2_csp/, profiles/agent_v2_csp_screen.json, docs/MEOWMIND_RL_SCREEN_SEEDS.md]
depends: [T0072]
land: direct
branch: antigravity/t0073-host-screen-agent-v2-with-per-count-p
updated: 2026-10-16
---

## Goal

Decide whether better opponent prediction wins games before investing in bigger P models. Paired host screen of agent_v2 with the T0067 per-count P bank (variant agent_v2_csp_direct, FUSION_TIER_CSP, from T0072 / PR 64) minus agent_v2 with the deployed XXL P (agent_v2_p2xxl_direct). Both actor-direct with the same actor; the only difference is the P bank.

Why (Casper, 2026-10-16): T0067 specialists beat the XXL P on held-out test loss by 0.04 (n14) to 0.17 (n02) nats, int16, all 14 intervals below 0. Earlier, T0032 (reports/models/p_distill/RESULTS.md) found an 8XL teacher trained on 24.15M rows beat XXL by only 0.028, and distilling it into XXL kept 0.0116 (42%), offline only, never played. T0071 (PR 63) showed a per-count 8XL from random init on the T0066 data (112k-840k train rows per count, 30-200x less than 24M) is worse than XXL at every count. No P loss gain has yet been shown to win games; if these larger gains do not, a further ~0.01 from distillation will not either.

Steps:
1. Reserve fresh seeds in docs/MEOWMIND_RL_SCREEN_SEEDS.md before any game: one preflight seed and one screen seed, never consumed ones.
2. Eight-worker preflight under lock cpu of agent_v2_csp_direct: counts 2/5/8/15, 16 maps each, held-out roster (profiles/meow_rl_screen.json), 2,000 ms first / 50 ms later replies. Record all-seat faults, first-reply startup with every served P model, p50/p95/p99/max. This is also the game-level host timing T0067 deferred. A failure is recorded and blocks the screen; never lower concurrency.
3. Only if it passes: paired screen agent_v2_csp_direct - agent_v2_p2xxl_direct, 28 maps per count 2-15 (392 fixtures), exactly 8 workers, A/A copy, 95% map-cluster interval, counts weighted equally, frozen before play, no early stopping. Per-count rows are descriptive.
4. Commit receipts and reports under reports/host/agent_v2_csp/, with the screen profile at profiles/agent_v2_csp_screen.json.

What follows from the result (not part of this task):
- helps: plan one general 8XL teacher on all data (the old 24M rows plus the T0066 rows), warm-started so it beats XXL, distilled into per-count XXL students, then screened the same way. Expect a small gain (about 0.01 nats per T0032).
- not shown or hurts: drop the 8XL/distillation route for P; better offline P loss is not the bottleneck.
- More matches per count are still worth collecting for T0068 RL (fresh on-policy games), not to make a per-count 8XL work.

## Done when

Seeds reserved before play; PREFLIGHT receipt (passed at 8 workers, or a blocked handoff); if it passed, the paired screen report with A/A and the overall verdict committed under reports/host/agent_v2_csp/

## Handoff

*2026-10-16 20:09Z antigravity@desktop-g60ojud:*

Seeds reserved (20261960 preflight, 20261961 screen). 8-worker preflight passed at 11.7s wall time with 0 faults and <9.4ms steady reply. Paired screen of 1,176 games completed at 132.3s wall time: overall difference -0.0116 [-0.0315, +0.0082] (not shown); A/A identical on 392/392 fixtures. Per task specification: offline P loss improvement does not yield in-game playing benefit, so the 8XL distillation path (T0074) is dropped/blocked, clearing the runway for T0068.

## Log

- 2026-10-16 19:31Z claude-review@caspers-macbook: created
- 2026-10-16 19:47Z antigravity@desktop-g60ojud: claimed
- 2026-10-16 20:08Z antigravity@desktop-g60ojud: note: eight-worker preflight pass; paired screen of 1,176 games complete; verdict not shown (-0.0116 [-0.0315, +0.0082]); A/A identical scores 392/392
- 2026-10-16 20:09Z antigravity@desktop-g60ojud: done
