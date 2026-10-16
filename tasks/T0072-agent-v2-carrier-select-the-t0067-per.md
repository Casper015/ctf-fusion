---
id: T0072
title: "agent_v2 carrier: select the T0067 per-count P bank (FUSION_TIER_CSP)"
status: review
owner: claude-csp@caspers-macbook
needs: []
writes: [bots/agent_v2/, inference/agent_v2/, tools/cpp/agent_v2/, training/agent_v2/, tests/test_agent_v2.py, tests/test_agent_v2_training.py, docs/AGENT_V2.md]
depends: []
land: pr
branch: claude-csp/t0072-agent-v2-carrier-select-the-t0067-per
updated: 2026-10-16
---

## Goal

Make the agent_v2 carrier able to load the per-count opponent P specialists from T0067 (inference/count_specialists_p/, namespaces csp_nXX) through a new tier FUSION_TIER_CSP, so T0068 can train per-count PPO specialists on that bank without editing carrier code. Existing tiers keep their behavior; the predictor's model identity names the bank; one bundled actor-direct variant agent_v2_csp_direct; training/agent_v2/bridge.py builds the bridge for the tier and records the bank in the run identity. Mac code and tests only; no host timing or strength games.

## Done when

PR open with CI green; the new variant bundles and compiles; a short replay on the stream_n02/n04/n15 fixtures runs without refusals; the bridge built with the new tier names the bank; existing agent_v2 tests pass; the PR says T0068 selects the bank with FUSION_TIER_CSP.

## Handoff

*2026-10-16 19:06Z claude-csp@caspers-macbook:*

PR https://github.com/Casper015/ctf-fusion/pull/64 (branch claude-csp/t0072-agent-v2-carrier-select-the-t0067-per, commits aa44e10 + cc1b027). Done: inference/agent_v2/p_bank.hpp maps FUSION_TIER_* to P's registry for bot, bridge and check; FUSION_TIER_CSP routes to fusion::models::count_specialists_p::forTeams (width 2048), model ids csp_n02..csp_n15, mr_p_bank names the library's bank; variant agent_v2_csp_direct (same actor as agent_v2_p2xxl_direct) registered in bots/agent_v2/variants.json, profiles/agent_v2.json unchanged; training.agent_v2 identity/bridge/train take --p-bank count_specialists_p and record bank, tier, width and every header hash, so cross-bank resume is refused. T0068 selects the bank with: python -m training.agent_v2.train --init-actor <actor .npz> --p-bank count_specialists_p ... Verified on the Mac (g++-16): agent_v2 + training tests 34 ok (4 skips: torch/engine), torch trainer tests 6 ok in a torch 2.14.1 venv, existing p2xxl_direct replies byte-identical to origin/main on n02/n04/n15. Next: CI green, then merge the PR. Not verified: Windows, any timing (first reply with the 2048-wide bank), strength.

## Log

- 2026-10-16 18:52Z claude-csp@caspers-macbook: created
- 2026-10-16 18:52Z claude-csp@caspers-macbook: amended depends -> none
- 2026-10-16 18:52Z claude-csp@caspers-macbook: note: dropped depends T0067: its PR 62 is merged into main (b0f5a3b), so inference/count_specialists_p/ is on main; T0067 only awaits its board done mark
- 2026-10-16 18:52Z claude-csp@caspers-macbook: claimed
- 2026-10-16 19:06Z claude-csp@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/64 open: FUSION_TIER_CSP + --p-bank count_specialists_p; T0068 selects the bank with it, no carrier edits
- 2026-10-16 19:06Z claude-csp@caspers-macbook: handoff -> review
