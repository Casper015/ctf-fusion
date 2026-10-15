---
id: T0052
title: "Register own_action_first_trial as a model study so main is green"
status: done
owner: claude-review@caspers-macbook
needs: []
writes: [tests/test_repo_layout.py, CONTRIBUTING.md]
depends: []
land: pr
branch: claude-review/t0052-register-own-action-first-trial-as-a
updated: 2026-10-15
---

## Goal

PR 49 (T0050) merged reports/models/own_action_first_trial/ without registering the folder, so tests.test_repo_layout.TestRepoLayout.test_reports_are_grouped fails on main in every test job. Add own_action_first_trial to MODEL_STUDIES in tests/test_repo_layout.py and to the study list in CONTRIBUTING.md. No other changes.

## Done when

PR merged; main CI green on test_repo_layout.

## Handoff

*2026-10-15 04:01Z claude-review@caspers-macbook:*

PR 50 merged (4c15169f): own_action_first_trial registered in MODEL_STUDIES and CONTRIBUTING.md; fixes test_reports_are_grouped on main.

## Log

- 2026-10-15 03:52Z claude-review@caspers-macbook: created
- 2026-10-15 03:52Z claude-review@caspers-macbook: claimed
- 2026-10-15 03:55Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/50 opened; layout test and core suite (326, 44 skipped) pass on macOS.
- 2026-10-15 04:01Z claude-review@caspers-macbook: done
