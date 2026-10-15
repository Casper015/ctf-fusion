---
id: T0008
title: "Triage the handoff prompts written before the board"
status: dropped
owner: ""
needs: []
writes: [docs/handoffs/, roster/report/NEXT_AGENT_PROMPT.md, reports/reviews/review_bundle_20261009/]
depends: []
land: direct
branch: ""
updated: 2026-10-11
---

## Goal

Three long prompts predate the board, and nobody has recorded which parts were done:
- [roster/report/NEXT_AGENT_PROMPT.md](../../roster/report/NEXT_AGENT_PROMPT.md)
- [reports/reviews/review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md](../../reports/reviews/review_bundle_20261009/NEXT_AGENT_EXECUTION_PROMPT.md)
- [docs/handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md](../../docs/handoffs/diverse15_distance_tiered_policy/NEXT_AGENT_PROMPT.md)

For each step in them, find whether it is done (cite the commit or report), still open, or obsolete.

## Done when

- Every still-open step is a task on the board (`task.py new`).
- Each prompt file starts with a line naming those tasks.
- Anything you cannot decide goes to Casper as a `blocked` question.

## Handoff

*2026-10-11 01:32Z claude-ci@caspers-macbook:*

Merged into T0024 (Board and docs consolidation: merge duplicate tasks, task summary, markdown digest and index) by claude-ci@caspers-macbook: the same documentation work: T0015's three doc edits and T0008's triage of the pre-board prompts are part of reading and summarising every markdown file

## Log

- 2026-10-10 08:52Z claude@caspers-macbook: created from the pre-board handoff prompts
- 2026-10-11 01:32Z claude-ci@caspers-macbook: merged into T0024