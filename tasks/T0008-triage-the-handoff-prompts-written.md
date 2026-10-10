---
id: T0008
title: "Triage the handoff prompts written before the board"
status: open
owner: ""
needs: []
writes: [docs/handoffs/, roster/report/NEXT_AGENT_PROMPT.md, reports/reviews/review_bundle_20261009/]
depends: []
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Three long prompts predate the board, and nobody has recorded which parts were done:
- [roster/report/NEXT_AGENT_PROMPT.md](../roster/report/NEXT_AGENT_PROMPT.md)
- [reports/reviews/review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md](../reports/reviews/review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md)
- [docs/handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md](../docs/handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md)

For each step in them, find whether it is done (cite the commit or report), still open, or obsolete.

## Done when

- Every still-open step is a task on the board (`task.py new`).
- Each prompt file starts with a line naming those tasks.
- Anything you cannot decide goes to Casper as a `blocked` question.

## Handoff

Not started. Read `git log --stat` around each prompt's date. Don't run anything: this is reading and bookkeeping.

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from the pre-board handoff prompts
