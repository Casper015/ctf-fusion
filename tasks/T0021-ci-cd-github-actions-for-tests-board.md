---
id: T0021
title: "CI/CD: GitHub Actions for tests, board check and bot bundle releases"
status: active
owner: claude-ci@caspers-macbook
needs: []
writes: [".github/", tools/repo/task.py, tools/repo/release_bots.py, tests/test_task_board.py, tests/test_release_bots.py, tests/test_repo_layout.py, tests/test_generate_roster_report.py, requirements-dev.txt, CONTRIBUTING.md, AGENTS.md, README.md, docs/TECHNICAL_OVERVIEW.md, tools/README.md, tasks/README.md]
depends: []
land: pr
branch: claude-ci/t0021-ci-cd-github-actions-for-tests-board
updated: 2026-10-10
---

## Goal

Run the checks on every push and pull request, and make the bots deliverable. CI: unit tests on Linux, macOS (NEON path) and Windows (the host OS), the P-only tests, the board and layout checks, optional torch parity and AI_Test pipeline tests. Delivery: a manual workflow that bundles every P-only variant into the single-file bot a judge compiles, syntax-checks it, and attaches it with SHA-256 hashes to a GitHub release.

## Done when

- The workflows run green on the PR.
- A clean checkout with only requirements-dev.txt passes the suite.
- CONTRIBUTING.md and AGENTS.md say what CI runs and what to do when main is red.

## Handoff

Not started.

## Log

- 2026-10-10 23:47Z claude@caspers-macbook: created
- 2026-10-10 23:48Z claude-ci@caspers-macbook: claimed
- 2026-10-10 23:48Z claude-ci@caspers-macbook: amended writes -> .github/, tools/repo/task.py, tools/repo/release_bots.py, tests/test_task_board.py, tests/test_release_bots.py, tests/test_repo_layout.py, tests/test_generate_roster_report.py, requirements-dev.txt, CONTRIBUTING.md, AGENTS.md, README.md, docs/TECHNICAL_OVERVIEW.md, tools/README.md, tasks/README.md
