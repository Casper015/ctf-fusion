---
id: T0072
title: "agent_v2 carrier: select the T0067 per-count P bank (FUSION_TIER_CSP)"
status: done
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

*2026-10-16 19:37Z claude-csp@caspers-macbook:*

merged as PR 64 (4c6321c): FUSION_TIER_CSP / --p-bank count_specialists_p selects T0067's per-count P specialists in the agent_v2 carrier; variant agent_v2_csp_direct

## Log

- 2026-10-16 18:52Z claude-csp@caspers-macbook: created
- 2026-10-16 18:52Z claude-csp@caspers-macbook: amended depends -> none
- 2026-10-16 18:52Z claude-csp@caspers-macbook: note: dropped depends T0067: its PR 62 is merged into main (b0f5a3b), so inference/count_specialists_p/ is on main; T0067 only awaits its board done mark
- 2026-10-16 18:52Z claude-csp@caspers-macbook: claimed
- 2026-10-16 19:06Z claude-csp@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/64 open: FUSION_TIER_CSP + --p-bank count_specialists_p; T0068 selects the bank with it, no carrier edits
- 2026-10-16 19:06Z claude-csp@caspers-macbook: handoff -> review
- 2026-10-16 19:37Z claude-csp@caspers-macbook: done
