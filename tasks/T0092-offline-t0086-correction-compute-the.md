---
id: T0092
title: "Offline T0086 correction: compute the gates from receipts, withdraw in-sample headroom claims"
status: open
owner: ""
needs: []
writes: [tools/search_diagnostics/summarize_pilot.py, tools/search_diagnostics/tests/test_summarize_pilot.py, reports/host/search_branch_diagnostics/REPORT.json, reports/host/search_branch_diagnostics/REPORT.md, reports/host/search_branch_diagnostics/CORRECTION.md]
depends: [T0086]
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

Offline correction of T0086 from its committed receipts only (no host, no new games). At 5c20cc4, tools/search_diagnostics/summarize_pilot.py writes the four decision gates, their rationales and the source/root/episode totals as literals, mean_of([]) returns 0, and REPORT.md is hand-written. Gate A ("candidate headroom exists") rests on in_sample_best_gain, which analyze.py itself defines as "biased upward, never a demonstrated gain"; T0086's goal prescribes disjoint-repeat selection and evaluation (cross_fit_gain), which is near zero with mixed signs and has no interval in any split/stratum group. Several statements go beyond the data ("mechanics mismatch completely ruled out", "P is not the root cause", "V estimation noise is the primary root cause", "V has virtually zero ranking ability" without an interval).

1. Rewrite summarize_pilot.py so every figure and every gate decision is computed from ANALYSIS_PRIMARY/EXTENDED, PARITY, P2/P3_EVAL, ESTIMATE, sources/ and reports/host/agent_v2_p3_search/EVIDENCE_AUDIT.json, with the gate rules stated in the code and the report and labelled as written after collection. Uncertainty uses analyze.aggregate (map clusters, equal counts); missing is unavailable, never zero; figures without a committed receipt are marked as such.
2. Render REPORT.md from REPORT.json alone; test that the committed pair is current.
3. CORRECTION.md: what is withdrawn and why, the original at 5c20cc4.
4. T0088 waits on this task; T0087 stays permitted only if the recomputed Gate D supports it.

## Done when

A reviewed PR: summarize_pilot.py computes every figure and gate from committed receipts and fails closed on missing inputs; REPORT.json/REPORT.md regenerated and tested for consistency; CORRECTION.md lists each withdrawn claim; tests cover gate rules, missing data and uncertainty-unavailable cases. T0088's status follows the recomputed Gate A.

## Handoff

Not started.

## Log

- 2026-10-18 17:22Z claude-fervent@cloud: created
