---
id: T0046
title: "Task board: archive finished tasks and a live board in tasks/README.md"
status: done
owner: claude@caspers-macbook
needs: []
writes: [tools/repo/task.py, tests/test_task_board.py, tests/test_repo_layout.py, tasks/README.md, AGENTS.md]
depends: []
land: pr
branch: claude/t0046-task-board-archive-finished-tasks-and-a
updated: 2026-10-14
---

## Goal

Casper's request (2026-10-14): move finished tasks into tasks/archive/ and make tasks/README.md show the current board whenever it is opened. task.py reads tasks/ and tasks/archive/, places every task by its status with each board change (done and dropped in archive/, everything else in tasks/), offers a one-time 'archive' command for the tasks finished before this change, and rewrites a generated board section in tasks/README.md (active, in review, blocked, ready, waiting, recently finished) with every board change, like SUMMARY.md. Links to a task file stay valid wherever the task sits (the link test treats tasks/<file> and tasks/archive/<file> as the same task). docs/MARKDOWN_INDEX.md is left to T0037's lane (stale until regenerated: a CI warning, not a failure).

## Done when

PR merged; the board tests cover placement on done/drop/merge, the archive command, the README section and the check warnings; after the merge 'task.py archive' has moved every finished task.

## Handoff

*2026-10-14 23:53Z claude@caspers-macbook:*

result: PR #43 merged and 'task.py archive' run: finished tasks live in tasks/archive/ and move there by themselves; tasks/README.md shows the live board.

## Log

- 2026-10-14 21:02Z claude@caspers-macbook: created
- 2026-10-14 21:02Z claude@caspers-macbook: claimed
- 2026-10-14 21:08Z claude@caspers-macbook: handoff -> review
- 2026-10-14 23:53Z claude@caspers-macbook: done
