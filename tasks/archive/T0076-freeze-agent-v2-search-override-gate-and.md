---
id: T0076
title: "Freeze agent_v2 search override gate and wider fixed-work diagnostics"
status: done
owner: claude-t76@caspers-macbook
needs: []
writes: [inference/agent_v2/search.hpp, bots/agent_v2/search_gate/, tools/cpp/agent_v2/check.cpp, tools/screens/search_gate_offline/, tests/test_agent_v2_search_gate.py, docs/AGENT_V2_SEARCH_GATE.md, docs/SEARCH_GATE_DIAGNOSTIC_SEEDS.md]
depends: []
land: pr
branch: claude-t76/t0076-freeze-agent-v2-search-override-gate-and
updated: 2026-10-17
---

## Goal

Repair agent_v2's bounded search offline before any host screen. T0064 found the old carrier's 2-ply V search lost to direct play (FP32 -3.91 pp, WideAct16 -4.66 pp) while overriding the actor on 73% of roots, 82% of them with a top-two margin under 0.01 (reports/host/meow_rl_aligned/diagnostics/DIAGNOSTICS.md); agent_v2's own search (default 4 candidates x 2 scenarios, inference/agent_v2/search.hpp) was never strength-screened. Add an override gate: replace the actor's joint only when mean V(best) - mean V(actor) > tau over identical completed scenarios; ties, incomplete work and refusals keep the complete actor order. Compare the narrow 4x2 tree with a wider 8x4 tree on pinned recorded states, with game-disjoint selection and validation splits. Freeze the numerical rule for choosing tau (from repeat-scenario stability and backend error) before running diagnostics, then freeze tau and breadth before any host data. Keep the existing actor, FP32 critic and XXL P. Reserve seed 20262120 (check first) only for offline scenario randomness, never for games. Report actor-relative margins, override rates, ranking stability and simulator checks. No played fixtures, no host timing, no strength claim; stop on parity or isolation failure, or if rankings are not stable enough to support any tau.

## Done when

PR merged with reproducible diagnostic receipts, the frozen tau and breadth, gate/fallback tests (ties, incomplete work and refusals keep the actor's order), and host-ready gated-wide and ungated-wide variants; the report separates numerical stability from any evidence about winning.

## Handoff

*2026-10-17 05:39Z claude-t76@caspers-macbook:*

Merged PR 67 as b3be3451c3 at Casper's request, with 6 CI jobs still pending (main CI checked after the merge). Frozen: 8 candidates x 4 scenarios, tau 0.002 (pre-registered rule, docs/AGENT_V2_SEARCH_GATE.md); n04 validation passed (C 0.754 [0.718, 0.788], G > 0); host variants agent_v2_p2xxl_gw8x4_fp32 / uw8x4_fp32 in bots/agent_v2/search_gate/. Self-consistency only, not strength. Risk for T0079: on the Mac most 8x4 trees did not complete (almost none at 15 teams), so its preflight must count completed trees per count. Closed by claude-review on behalf of the deleted claude-t76 session.

## Log

- 2026-10-17 04:06Z claude-review@caspers-macbook: created
- 2026-10-17 04:21Z claude-t76@caspers-macbook: claimed
- 2026-10-17 05:28Z claude-t76@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/67: frozen W 8x4, tau 0.002; validation passed; GW/UW variants
- 2026-10-17 05:28Z claude-t76@caspers-macbook: handoff -> review
- 2026-10-17 05:39Z claude-t76@caspers-macbook: done
