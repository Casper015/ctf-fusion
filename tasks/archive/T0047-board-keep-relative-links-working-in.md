---
id: T0047
title: "Board: keep relative links working in archived task files"
status: done
owner: claude@caspers-macbook
needs: []
writes: [tools/repo/task.py, tests/test_task_board.py]
depends: []
land: pr
branch: claude/t0047-board-keep-relative-links-working-in
updated: 2026-10-15
---

## Goal

T0046 regression: task files moved into tasks/archive/ keep links written relative to tasks/ (../bots/..., README.md), which now point one folder too high; tests/test_repo_layout's link check fails on the next code push. task.py must rewrite a moved file's broken relative links to the same targets from its new folder (and back when a task is reopened), and 'task.py archive' must repair the files already archived. Links that already resolve, URLs, anchors and task-file links are left alone.

## Done when

PR merged; the archive repair commit run; test_markdown_links_resolve passes on main.

## Handoff

*2026-10-15 00:59Z claude@caspers-macbook:*

result: PR #45 merged: task.py keeps relative links working when task files move between tasks/ and tasks/archive/; archived files repaired on main.

## Log

- 2026-10-15 00:10Z claude@caspers-macbook: created
- 2026-10-15 00:10Z claude@caspers-macbook: claimed
- 2026-10-15 00:39Z claude@caspers-macbook: handoff -> review
- 2026-10-15 00:59Z claude@caspers-macbook: done
