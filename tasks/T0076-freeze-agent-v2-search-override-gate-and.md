---
id: T0076
title: "Freeze agent_v2 search override gate and wider fixed-work diagnostics"
status: open
owner: ""
needs: []
writes: [inference/agent_v2/search.hpp, bots/agent_v2/search_gate/, tools/cpp/agent_v2/check.cpp, tools/screens/search_gate_offline/, tests/test_agent_v2_search_gate.py, docs/AGENT_V2_SEARCH_GATE.md, docs/SEARCH_GATE_DIAGNOSTIC_SEEDS.md]
depends: []
land: pr
branch: ""
updated: 2026-10-17
---

## Goal

Repair agent_v2's bounded search offline before any host screen. T0064 found the old carrier's 2-ply V search lost to direct play (FP32 -3.91 pp, WideAct16 -4.66 pp) while overriding the actor on 73% of roots, 82% of them with a top-two margin under 0.01 (reports/host/meow_rl_aligned/diagnostics/DIAGNOSTICS.md); agent_v2's own search (default 4 candidates x 2 scenarios, inference/agent_v2/search.hpp) was never strength-screened. Add an override gate: replace the actor's joint only when mean V(best) - mean V(actor) > tau over identical completed scenarios; ties, incomplete work and refusals keep the complete actor order. Compare the narrow 4x2 tree with a wider 8x4 tree on pinned recorded states, with game-disjoint selection and validation splits. Freeze the numerical rule for choosing tau (from repeat-scenario stability and backend error) before running diagnostics, then freeze tau and breadth before any host data. Keep the existing actor, FP32 critic and XXL P. Reserve seed 20262120 (check first) only for offline scenario randomness, never for games. Report actor-relative margins, override rates, ranking stability and simulator checks. No played fixtures, no host timing, no strength claim; stop on parity or isolation failure, or if rankings are not stable enough to support any tau.

## Done when

PR merged with reproducible diagnostic receipts, the frozen tau and breadth, gate/fallback tests (ties, incomplete work and refusals keep the actor's order), and host-ready gated-wide and ungated-wide variants; the report separates numerical stability from any evidence about winning.

## Handoff

Not started.

## Log

- 2026-10-17 04:06Z claude-review@caspers-macbook: created
