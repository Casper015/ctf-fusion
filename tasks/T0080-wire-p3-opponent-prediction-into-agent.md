---
id: T0080
title: "Wire P3 opponent prediction into agent_v2 search"
status: active
owner: codex-p3search@caspers-macbook
needs: []
writes: [bots/agent_v2/p3_search/, inference/agent_v2/p_bank.hpp, inference/agent_v2/predictor.hpp, tests/test_agent_v2_p3.py, tools/cpp/agent_v2/check.cpp, tools/screens/next_controls/MANIFEST.json]
depends: [T0076]
land: pr
branch: codex-p3search/t0080-wire-p3-opponent-prediction-into-agent
updated: 2026-10-17
---

## Goal

Casper explicitly requested replacing P2 opponent prediction with P3 to test search strength. Add the existing P3 XXL bank and full P3 feature/conditional action/target path, including correct root and branch rules/history, without mixing legacy forecasts. Keep frozen actor, critic, resolver, candidate/scenario counts, threshold and deadline identical to matched P2 controls. Provide separate direct and gated-search variants and a P3-vs-P2 fixed-work harness. Preserve T0079 inputs/running jobs; do not promote defaults or start host games here.

## Done when

Native P3 outputs equal the existing P3 implementation on root and branch states; rules from INIT initialized per view; masks, targets, pure-model identity and branch isolation pass. Actual P3 direct/search variant sources compile and replay protocol fixtures. Existing P2/CSP behavior stays compatible. Register host follow-up with exactly eight workers, full 2s/50ms limits, completed-search telemetry and paired fresh seeds. Deliver PR with code/tests and truthful hardware limitations.

## Handoff

Not started.

## Log

- 2026-10-17 07:46Z codex-p3search@caspers-macbook: created
- 2026-10-17 07:46Z codex-p3search@caspers-macbook: claimed
- 2026-10-17 07:50Z codex-p3search@caspers-macbook: amended writes -> bots/agent_v2/p3_search/, inference/agent_v2/p_bank.hpp, inference/agent_v2/predictor.hpp, tests/test_agent_v2_p3.py, tools/cpp/agent_v2/check.cpp, tools/screens/next_controls/MANIFEST.json
