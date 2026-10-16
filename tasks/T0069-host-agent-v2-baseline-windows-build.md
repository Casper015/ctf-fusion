---
id: T0069
title: "Host: agent_v2 baseline (Windows build/tests, eight-worker preflight, paired screen vs meow_rl_aligned_p2_large)"
status: active
owner: antigravity@desktop-g60ojud
needs: [host, ai-test]
writes: [docs/MEOWMIND_RL_SCREEN_SEEDS.md, reports/host/agent_v2/, reports/host/agent_v2/, docs/MEOWMIND_RL_SCREEN_SEEDS.md, bots/agent_v2/]
depends: [T0065]
land: direct
branch: antigravity/t0069-host-agent-v2-baseline-windows-build
updated: 2026-10-16
---

## Goal

Establish the agent_v2 carrier's host baseline (T0065, docs/AGENT_V2.md, profiles/agent_v2.json) on desktop-g60ojud before any fresh collection or training on it. No training.
1. Build bots/agent_v2/variants.json with python bots/p_only/tools/build.py --variants bots/agent_v2/variants.json. Then run tests/test_agent_v2.py and tests/test_agent_v2_training.py there, with AI_TEST_ROOT and torch, and record the results, including the AVX2 kernels and the torch smoke/resume tests that skip on the Mac.
2. Reserve fresh seeds in docs/MEOWMIND_RL_SCREEN_SEEDS.md before any game: one preflight seed and one screen seed, never consumed ones.
3. Run an eight-worker preflight under "lock cpu" of agent_v2_p2xxl_direct, plus agent_v2_p2xxl_search_w16 as a timing-only experimental arm.
   - Counts 2/5/8/15, 16 maps each, the held-out roster, 2,000 ms first and 50 ms later full replies.
   - Record all-seat faults, counter coverage, first-reply startup with every served P model, p50/p95/p99/max.
   - A failure is recorded and blocks the screen. Never lower the concurrency.
4. Only if the preflight passes, run a paired screen of agent_v2_p2xxl_direct - meow_rl_aligned_p2_large.
   - 28 maps per count 2-15 (392 fixtures), exactly 8 workers, 95% map-cluster interval, A/A copy of agent_v2.
   - Frozen before play, with no early stopping.
   - Report it as a carrier comparison: P, resolver and simulator change together, so it is not a single-factor result.
5. Commit the receipts and reports under reports/host/agent_v2/. Search arms are not screened here: T0064 measured that search hurts with every critic, and its deficit must be diagnosed first.

## Done when

test results, PREFLIGHT receipt (passed at 8 workers or a blocked handoff) and, if it passed, the paired screen report committed under reports/host/agent_v2/

## Handoff

Not started.

## Log

- 2026-10-16 01:47Z claude-review@caspers-macbook: created
- 2026-10-16 01:47Z claude-review@caspers-macbook: amended goal
- 2026-10-16 03:35Z antigravity@desktop-g60ojud: claimed
- 2026-10-16 03:49Z antigravity@desktop-g60ojud: amended writes -> docs/MEOWMIND_RL_SCREEN_SEEDS.md, reports/host/agent_v2/, reports/host/agent_v2/,docs/MEOWMIND_RL_SCREEN_SEEDS.md,bots/agent_v2/
- 2026-10-16 04:17Z antigravity@desktop-g60ojud: landed: 159270127b on main, 13 file(s)
