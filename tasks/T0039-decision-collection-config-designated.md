---
id: T0039
title: "Decision collection config: designated log carrier, decision_log never an opponent (follow-up to PR #33)"
status: done
owner: claude-ci@caspers-macbook
needs: []
writes: [profiles/decision_opponents.json, profiles/decision_variants.json, tests/test_decision_opponents.py]
depends: [T0035, T0036]
land: pr
branch: claude-ci/t0039-decision-collection-config-designated
updated: 2026-10-12
---

## Goal

Follow-up to PR #33 (T0036), which merged before the configuration switch. Make v12_ponly_xxl_dlog (the log-only build of the admitted v12_ponly_xxl/P2 incumbent, PONLY_DECISION 1) the designated carrier of profiles/decision_opponents.json; add bots/p_only/profiles/variants_decision.json to the catalogs of profiles/decision_variants.json and exclude role decision_log so log carriers are never opponents; update tests. Code/config only: no build, game or seed.

## Done when

PR merged; the registry resolves the designated log carrier, lists both log carriers as not registered, and its tests pass.

## Handoff

*2026-10-12 16:30Z claude-ci@caspers-macbook:*

Merged as PR #35 (2026-10-12 16:24Z): designated carrier v12_ponly_xxl_dlog, decision_log excluded from opponents.

## Log

- 2026-10-12 16:03Z claude-ci@caspers-macbook: created
- 2026-10-12 16:03Z claude-ci@caspers-macbook: claimed
- 2026-10-12 16:06Z claude-ci@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/35 (follow-up to #33; linked there)
- 2026-10-12 16:06Z claude-ci@caspers-macbook: handoff -> review
- 2026-10-12 16:30Z claude-ci@caspers-macbook: done
