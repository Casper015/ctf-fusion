---
id: T0076
title: "Freeze agent_v2 search override gate and wider fixed-work diagnostics"
status: review
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

*2026-10-17 05:28Z claude-t76@caspers-macbook:*

In review: PR https://github.com/Casper015/ctf-fusion/pull/67 (branch claude-t76/t0076-freeze-agent-v2-search-override-gate-and).

Done:
- 885fda2: gate (`gated_choice`, AGENT_V2_SEARCH_TAU, default 0 = old behaviour for finite means), plus the pre-registered rule in docs/AGENT_V2_SEARCH_GATE.md section 1, written before any margin was computed. Seed 20262120 reserved in docs/SEARCH_GATE_DIAGNOSTIC_SEEDS.md.
- fb273c9, 01a6abe: `check gate` / `check gatetree`, tools/screens/search_gate_offline/{run,analyze}.py, bots/agent_v2/search_gate/bot.hpp, tests/test_agent_v2_search_gate.py. Codex review fixes: a non-finite mean anywhere keeps the actor; sums are sequential, like Search::choose; isolation is checked at both breadths.
- a08a415: frozen W = 8 candidates x 4 scenarios, tau 0.002.
  - selection on n02 + n15 (every seat); validation on n04 passed: C 0.754 [0.718, 0.788], G > 0;
  - eps 2.9e-7; parity bit-exact; isolation ok;
  - receipts in bots/agent_v2/search_gate/diagnostics/ (TREES sha256 57edea1a...).
- Variants agent_v2_p2xxl_gw8x4_fp32 (tau 0.002) and agent_v2_p2xxl_uw8x4_fp32 (tau 0); bots/agent_v2/search_gate/variants.json includes D.

Tests:
- tests/test_agent_v2_search_gate.py: 12 ok.
- tests/test_agent_v2.py: 21 ok, before merging main.
- The full discover after merging main was stopped by request; CI covers it.

Reproduce: `python3 tools/screens/search_gate_offline/run.py --jobs 8 && python3 tools/screens/search_gate_offline/analyze.py`. The analyze step reproduces byte-identically.

Not shown / uncertain:
- This is numerical self-consistency under the search's own simulator, P and critic. It says nothing about winning.
- The corpus is 3 games, and the bootstrap resamples states, so the intervals are optimistic.
- The gate is mild: 68% of roots are still overridden, against 78% ungated.

RISK for T0079: admission settings are unchanged. Each trajectory must be admitted before 16.5 ms of the 30 ms deadline, and the worst-seen cost never decays. On the Mac (not evidence) most 8x4 trees were incomplete, and almost none at 15 teams. If the host is similar, GW and UW play D's order. T0079's preflight must report search_used per arm and count. An agent_v2 search admission calibration task on the host may be needed first.

## Log

- 2026-10-17 04:06Z claude-review@caspers-macbook: created
- 2026-10-17 04:21Z claude-t76@caspers-macbook: claimed
- 2026-10-17 05:28Z claude-t76@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/67: frozen W 8x4, tau 0.002; validation passed; GW/UW variants
- 2026-10-17 05:28Z claude-t76@caspers-macbook: handoff -> review
