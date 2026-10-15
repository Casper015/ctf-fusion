---
id: T0047
title: "Board: keep relative links working in archived task files"
status: review
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

*2026-10-15 00:39Z claude@caspers-macbook:*

PR #45 (CI green). Done: task.py rewrites a moved task file's broken relative links (tasks/ <-> tasks/archive/), archive repairs old files; already run on main (fixed T0003, T0004, T0008, T0024, T0025). Next: merge #45; nothing else to run.

## Log

- 2026-10-15 00:10Z claude@caspers-macbook: created
- 2026-10-15 00:10Z claude@caspers-macbook: claimed
- 2026-10-15 00:39Z claude@caspers-macbook: handoff -> review
