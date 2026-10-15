---
id: T0052
title: "Register own_action_first_trial as a model study so main is green"
status: open
owner: ""
needs: []
writes: [tests/test_repo_layout.py, CONTRIBUTING.md]
depends: []
land: pr
branch: ""
updated: 2026-10-15
---

## Goal

PR 49 (T0050) merged reports/models/own_action_first_trial/ without registering the folder, so tests.test_repo_layout.TestRepoLayout.test_reports_are_grouped fails on main in every test job. Add own_action_first_trial to MODEL_STUDIES in tests/test_repo_layout.py and to the study list in CONTRIBUTING.md. No other changes.

## Done when

PR merged; main CI green on test_repo_layout.

## Handoff

Not started.

## Log

- 2026-10-15 03:52Z claude-review@caspers-macbook: created
