---
id: T0021
title: "CI/CD: GitHub Actions for tests, board check and bot bundle releases"
status: review
owner: claude-ci@caspers-macbook
needs: []
writes: [".github/", ".gitignore", AGENTS.md, CONTRIBUTING.md, README.md, docs/TECHNICAL_OVERVIEW.md, requirements-dev.txt, tasks/README.md, tests/test_generate_roster_report.py, tests/test_release_bots.py, tests/test_repo_layout.py, tests/test_task_board.py, tools/README.md, tools/repo/release_bots.py, tools/repo/task.py]
depends: []
land: pr
branch: claude-ci/t0021-ci-cd-github-actions-for-tests-board
updated: 2026-10-11
---

## Goal

Run the checks on every push and pull request, and make the bots deliverable. CI: unit tests on Linux, macOS (NEON path) and Windows (the host OS), the P-only tests, the board and layout checks, optional torch parity and AI_Test pipeline tests. Delivery: a manual workflow that bundles every P-only variant into the single-file bot a judge compiles, syntax-checks it, and attaches it with SHA-256 hashes to a GitHub release.

## Done when

- The workflows run green on the PR.
- A clean checkout with only requirements-dev.txt passes the suite.
- CONTRIBUTING.md and AGENTS.md say what CI runs and what to do when main is red.

## Handoff

*2026-10-11 00:21Z claude-ci@caspers-macbook:*

PR https://github.com/Casper015/ctf-fusion/pull/12, all 11 checks green (Linux 3.10 and 3.13, macOS, Windows, P-only, torch, bundles, layout, board).

Not exercised until the PR is merged: the *publish* step of bots.yml (a workflow_dispatch run from main creates the release). Packaging, -fsyntax-only of all 29 bundles, SHA256SUMS check and artifact upload did run on the PR.

Needs Casper: merge; add secret AI_TEST_TOKEN (fine-grained, read-only on Casper015/AI_Test, which is private) to switch on the pipeline job; decide on branch protection (task.py land pushes to main directly, so required checks would block it). After merge: remove the worktree ../ctf-fusion.worktrees/T0021.

## Log

- 2026-10-10 23:47Z claude@caspers-macbook: created
- 2026-10-10 23:48Z claude-ci@caspers-macbook: claimed
- 2026-10-10 23:48Z claude-ci@caspers-macbook: amended writes -> .github/, tools/repo/task.py, tools/repo/release_bots.py, tests/test_task_board.py, tests/test_release_bots.py, tests/test_repo_layout.py, tests/test_generate_roster_report.py, requirements-dev.txt, CONTRIBUTING.md, AGENTS.md, README.md, docs/TECHNICAL_OVERVIEW.md, tools/README.md, tasks/README.md
- 2026-10-11 00:01Z claude-ci@caspers-macbook: amended writes -> .github/, .gitignore, AGENTS.md, CONTRIBUTING.md, README.md, docs/TECHNICAL_OVERVIEW.md, requirements-dev.txt, tasks/README.md, tests/test_generate_roster_report.py, tests/test_release_bots.py, tests/test_repo_layout.py, tests/test_task_board.py, tools/README.md, tools/repo/release_bots.py, tools/repo/task.py
- 2026-10-11 00:21Z claude-ci@caspers-macbook: handoff -> review
