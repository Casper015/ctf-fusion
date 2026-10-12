---
id: T0039
title: "Decision collection config: designated log carrier, decision_log never an opponent (follow-up to PR #33)"
status: open
owner: ""
needs: []
writes: [profiles/decision_opponents.json, profiles/decision_variants.json, tests/test_decision_opponents.py]
depends: [T0035, T0036]
land: pr
branch: ""
updated: 2026-10-12
---

## Goal

Follow-up to PR #33 (T0036), which merged before the configuration switch. Make v12_ponly_xxl_dlog (the log-only build of the admitted v12_ponly_xxl/P2 incumbent, PONLY_DECISION 1) the designated carrier of profiles/decision_opponents.json; add bots/p_only/profiles/variants_decision.json to the catalogs of profiles/decision_variants.json and exclude role decision_log so log carriers are never opponents; update tests. Code/config only: no build, game or seed.

## Done when

PR merged; the registry resolves the designated log carrier, lists both log carriers as not registered, and its tests pass.

## Handoff

Not started.

## Log

- 2026-10-12 16:03Z claude-ci@caspers-macbook: created
