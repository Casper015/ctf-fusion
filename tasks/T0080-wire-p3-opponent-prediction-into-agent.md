---
id: T0080
title: "Wire P3 opponent prediction into agent_v2 search"
status: review
owner: codex-p3search@caspers-macbook
needs: []
writes: [bots/agent_v2/p3_search/, inference/agent_v2/p_bank.hpp, inference/agent_v2/predictor.hpp, tests/test_agent_v2_p3.py, tools/cpp/agent_v2/check.cpp, tools/screens/next_controls/MANIFEST.json]
depends: [T0076]
land: pr
branch: codex-p3search/t0080-wire-p3-opponent-prediction-into-agent
updated: 2026-10-17
---

## Goal

Main comparison clarified by Casper: use the GENERAL P2 XXL (inference/models_pf_p2xxl_vm, pf_p2_xxl_s1) versus GENERAL P3 XXL (inference/models_pf_p3xxl_vm, pf_p3_xxl_s1) from the same T0011 feature-family study. Both serve one general policy across counts 2-15. Do not use the count-routed P2 bank as the main control; keep T0079 only as an external deployment reference. Pin both manifests, headers, feature-family and calibration identities. Own actor, FP32 critic, resolver, breadth, scenario keys, tau and deadline remain fixed. This clarification supersedes any older routing description below.

Casper explicitly requested replacing P2 opponent prediction with P3 to test search strength. Add the existing P3 XXL bank and full P3 feature/conditional action/target path, including correct root and branch rules/history, without mixing legacy forecasts. Keep frozen actor, critic, resolver, candidate/scenario counts, threshold and deadline identical to matched P2 controls. Provide separate direct and gated-search variants and a P3-vs-P2 fixed-work harness. Preserve T0079 inputs/running jobs; do not promote defaults or start host games here.

## Done when

Native P3 outputs equal the existing P3 implementation on root and branch states; rules from INIT initialized per view; masks, targets, pure-model identity and branch isolation pass. Actual P3 direct/search variant sources compile and replay protocol fixtures. Existing P2/CSP behavior stays compatible. Register host follow-up with exactly eight workers, full 2s/50ms limits, completed-search telemetry and paired fresh seeds. Deliver PR with code/tests and truthful hardware limitations.

## Handoff

*2026-10-17 08:08Z codex-p3search@caspers-macbook:*

PR #68: https://github.com/Casper015/ctf-fusion/pull/68 ; latest commit 363b282e52566efc46283c452b1e728b42ec4870. User correction applied: main control is general P2 XXL (models_pf_p2xxl_vm) vs general P3 XXL (models_pf_p3xxl_vm), both T0011 seed1 at every count; routed P2 remains only T0079 external reference. Both root and branch P3 use its full feature/action/target implementation with INIT rules. Four matched direct/search sources, fixed-work CLI, native tests and regenerated source-bound control manifest delivered. Local checks: 474 core tests OK (60 skipped); 64 P-only tests OK (3 skipped); final seven focused P3 tests OK; actual four variants compiled/replayed and general P2/P3 fixed-work CLI smoke completed. The final test-only commit adds Windows line-ending/executable handling; local hardware was Mac, Windows/host timing remains unverified. GitHub Bots workflow succeeded when checked; CI still in progress. No source defaults or T0079 frozen inputs changed and no host games/training started. T0081 is the separate 8-worker host comparison after this PR merges and T0079 finishes; it freezes settings and reports completed-search work, cannot assume P3 helps. Existing own actor/critic stay T0059 aligned controls, not G; PPO P3 allowlist remains out of scope.

## Log

- 2026-10-17 07:46Z codex-p3search@caspers-macbook: created
- 2026-10-17 07:46Z codex-p3search@caspers-macbook: claimed
- 2026-10-17 07:50Z codex-p3search@caspers-macbook: amended writes -> bots/agent_v2/p3_search/, inference/agent_v2/p_bank.hpp, inference/agent_v2/predictor.hpp, tests/test_agent_v2_p3.py, tools/cpp/agent_v2/check.cpp, tools/screens/next_controls/MANIFEST.json
- 2026-10-17 07:56Z codex-p3search@caspers-macbook: amended goal
- 2026-10-17 08:08Z codex-p3search@caspers-macbook: handoff -> review
