---
id: T0075
title: "Freeze agent_v2 RL controls and original Meowmind search ablation"
status: active
owner: claude-t75@caspers-macbook
needs: []
writes: [bots/agent_v2/controls/, inference/agent_v2_controls/, bots/meowmind_p/attribution/, tools/screens/next_controls/, tests/test_next_controls.py]
depends: [T0068]
land: pr
branch: claude-t75/t0075-freeze-agent-v2-rl-controls-and-original
updated: 2026-10-17
---

## Goal

Build the controls that the T0068 evaluation and the Meowmind gap attribution need. (1) agent_v2 variants with the CSP P bank (FUSION_TIER_CSP) that play only T0068's general control (rl_general) on every count, and only the common initialization actor (data/count_specialists_rl/init_actor.npz, exported as INT16 with the same scheme, temperature, resolver and precision as agent_v2_rl_csp). Recover and record the init actor's sha256 and origin; do not assume another warm actor is equivalent, and stop with a blocked handoff if its provenance cannot be established. Use a dedicated variant manifest under bots/agent_v2/controls/ (T0068 reserves bots/agent_v2/variants.json). (2) Original Meowmind wrappers that differ only in NN_SEARCH=1 versus NN_SEARCH=0 (bots/meowmind_p/meowmind-ctf-374-20261011.cpp line ~680), source and predictor untouched. (3) A preflight adapter for both telemetry formats that never fabricates absent counters. No games, no seed consumption, no timing or strength claims: tests replay the recorded streams only (bots/p_only/tests/fixtures/stream_n02/n04/n15).

## Done when

PR merged: every control compiles in CI; tests build the real variant files and replay n02/n04/n15 with valid orders, check that the general-only variant routes every count to rl_general and the init-only variant to the init actor with matching outExp, and check INT16 agreement of the init export; a manifest pins every arm's source, model and bundle hashes and shows that the two Meowmind wrappers differ only in NN_SEARCH.

## Handoff

Not started.

## Log

- 2026-10-17 04:06Z claude-review@caspers-macbook: created
- 2026-10-17 04:41Z claude-t75@caspers-macbook: claimed
- 2026-10-17 05:13Z claude-t75@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/66: G, ON/OFF, manifest, preflight adapter and tests done; I awaits the host export of init_actor.npz (sha256 not in T0068's committed records)
