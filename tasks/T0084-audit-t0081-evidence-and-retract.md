---
id: T0084
title: "Audit T0081 evidence and retract uncomputed search diagnostics"
status: open
owner: ""
needs: [host, ai-test]
writes: [tools/screens/p3_search_host/, reports/host/agent_v2_p3_search/]
depends: [T0081]
land: pr
branch: ""
updated: 2026-10-18
---

## Goal

Priority 0. Audit the committed T0081 results before any training decision. At main commit 903947b, tools/screens/p3_search_host/diagnose.py returns literal candidate headroom +0.042, P NLL 2.148/2.085, combat precision, Spearman 0.284, override loss fraction 0.608 and gain -0.0168 without reading corresponding observations or continuation returns. check_simulator_parity loads an engine and counts stream blocks but never compares native and engine transitions; parity rates are literal 1.0. Compute metrics have hard-coded fallbacks, and the Markdown/root-cause synthesis remains literal even when preflight data is supplied. These are NOT measurements and cannot establish V as the unique cause or exonerate candidates/P/simulator.

1. Preserve the original commit and raw artifacts. Create an evidence inventory mapping each claim to input hashes, extraction command, actual calculation and sample unit. Replace unimplemented diagnostic outputs with explicit unavailable statuses and reasons; missing/corrupt input must not produce measured success. Generated Markdown/JSON must agree and contain only computed metrics or clearly labeled hypotheses. Add regression checks for missing inputs, changed inputs and impossible success without execution.
2. Regenerate/correct derived reports with an explicit correction/provenance record; do not silently replace original measurements or manually edit generated metrics. Keep verified match results distinct from unsupported causal assertions. Existing older simulator tests can be cited at their exact source and coverage, never passed off as a new T0081 audit.
3. On the host independently reaggregate the 3920-game schedule and all-seat logs: paired utilities, map clusters/equal-count weighting, three-contrast Bonferroni intervals, frozen binary/profile identities, missingness, actual seeds, interleaving and roster overlap with both P training corpora. Check whether committed fixed-work receipts actually ran the native harness. If absent, label missing; no fabricated replacement or automatic new games.
4. Investigate formal-screen failures: committed SCREEN.md reports candidate faults 2/3/4 in P2-search/P3-search/P3-search-AA; P3 has 3 late replies and AA has 2, and P2-direct opponents have 2 faults/late replies. Reconcile reply timing conventions and fault reasons. Preserve the scheduled intention-to-treat result; any paired fault-free sensitivity is secondary and cannot substitute for it.
5. Recompute A/A (-0.0091 utility reported; 456/784 identical score vectors) and distinguish wall-clock admission variability from RNG differences. Verify preflight search completion per count, not only the aggregate nonzero check in verdict(). Audit absent override/gate fields and missing timing handling. Do not silently fix the original experimental conditions or relabel not-shown as equivalence.
No new games or training. If raw host records are unavailable, deliver the correction and state precisely which screen claims remain unaudited.

Shared execution contract: this continues an architecture redesign, not a legacy-behavior refactor. Keep official rules exact; do not require whole-bot parity with Meowmind. Preserve agent_v2_p2xxl_direct as production default. P (opponent prediction), pi (our actor), and V/advantage scorer are distinct components. Follow AGENTS.md, claim with your own identity, respect active lanes (especially T0082/PR69 and T0051), and use isolated worktrees. All new timing, collection and strength evidence comes from desktop-g60ojud with exactly eight game workers under the CPU lease, GPU training stopped, 2000/50 ms complete replies. GPU training uses the GPU lease and is scheduled separately. Pre-register fresh map/flag/actor/scenario seeds and check derived-seed collisions; never replay consumed seeds as new evidence. Offline reanalysis of existing immutable artifacts is allowed. Freeze source, engine, model, runner, binary and profile hashes. Missing evidence is unavailable, never zero or a successful check. Do not extend budgets after inspecting outcomes. No default promotion in these tasks; no new giant P/V, CNN, count specialists, or full learned attack policy. All repository prose is English.


## Done when

A reviewed PR provides a reproducible evidence audit and corrected generated reports. Uncomputed constants cannot appear as experimental results; independent screen recomputation and fault/A-A accounting either reproduce the screen or explicitly qualify it. Every unsupported root-cause claim is withdrawn. Original artifacts remain traceable. Deliver a clear list of usable evidence and missing inputs for the diagnostic tasks.

## Handoff

Not started.

## Log

- 2026-10-18 03:35Z codex-search-plan@caspers-macbook: created
