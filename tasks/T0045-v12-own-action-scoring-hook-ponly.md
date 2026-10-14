---
id: T0045
title: "v12 own-action scoring hook (PONLY_DECISION 2), default off"
status: open
owner: ""
needs: []
writes: [bots/p_only/include/decision_score.hpp, bots/p_only/tools/derive/make_scoring.py, bots/p_only/bots/v12_ponly_fs.hpp, bots/p_only/tests/test_decision_scoring.py, bots/p_only/tests/replay_test.cpp, docs/OWN_ACTION_RUNTIME.md, CONTRIBUTING.md]
depends: [T0044]
land: pr
branch: ""
updated: 2026-10-14
---

## Goal

Stage 4 of Casper's own-action scorer preparation (2026-10-14). A generated scoring carrier v12_ponly_fs.hpp (from v12_ponly_f.hpp, which stays byte-identical, as do decision_hook.hpp and every file T0037's frozen identities hash) with PONLY_DECISION 2: score the carrier's actual shortlisted complete orders with their deciding carrier scores exactly as training/decision_branch_collect selects them, keep the incumbent, canonicalize and deduplicate as Python, apply training/evaluate_decisions.apply_rule (incumbent anchoring, zscore or incumbent_override, ties, unsupported and missing scores), fall back to the incumbent for an invalid model, a feature or identity mismatch or the deadline, report coverage counters, and compile to nothing when off or lambda = 0. Fixed models and recorded inputs only; C08 later. Host handoff for export parity, 50 ms validation and an equal-time paired screen.

## Done when

PR merged; off and lambda=0 builds reply byte-identically to the production family carrier on recorded games; the scoring build's choices equal Python's rule on the same shortlist; fallbacks and counters tested; host handoff documented.

## Handoff

Not started.

## Log

- 2026-10-14 20:03Z claude@caspers-macbook: created
