---
id: T0027
title: "Decide which pre-board evidence steps are still wanted"
status: blocked
owner: claude-ci@caspers-macbook
needs: []
writes: [docs/DIGEST.md]
depends: []
land: direct
branch: ""
updated: 2026-10-11
---

## Goal

The triage of the three pre-board prompts (T0024; the triage section of docs/DIGEST.md) left four steps that cost a lot of compute and that no current task needs. Whether to run them is Casper's call; the question is in Handoff.

## Done when

Casper has answered in a note. Each 'yes' is a task on the board, and the answers are recorded in docs/DIGEST.md (Triage of the pre-board prompts).

## Handoff

*2026-10-11 01:51Z claude-ci@caspers-macbook:*

Question for Casper. Four steps from the pre-board prompts are not done and no task needs them. Should each one run? My recommendation is in brackets.

1. **Full strength screen of every eligible roster bot** (roster prompt, step B5). Only the 64-bot light screen ran. All 258 bots at 50 maps per count is about 180,600 games, 10-12 host hours. [Drop: training_selection_20261009 chose the pool from the light screen. Confirm only the bots whose strength claims matter.]
2. **10,000 independent map groups per evaluated opponent bot** (diverse15 prompt, Phase A step 4). [Drop: the 10,000-group confirmation is for our finalists (T0006), not for every opponent.]
3. **Validate the opponent selection at equal training budget**: file-uniform, lineage-balanced, behavior-balanced and fixed-15 pools (diverse15 prompt, Phase A step 10). This means four training runs. [Defer until T0006 has finalists.]
4. **Fixed-point softmax and probability kernels** (docs/handoffs/diverse15_distance_tiered_policy/FIXED_POINT_PROBABILITY_PLAN.md). [Defer: at 15 teams under 16-seat load, XXL inference takes 2.74-2.83 ms of its 5 ms budget.]

Answer with `python tools/repo/task.py --as casper@mac note T0027 "Answer: 1 drop, 2 drop, 3 defer, 4 defer"`. Then whoever picks this up creates a task for each yes, records the answers in docs/DIGEST.md, and closes this task.

T0001 asks a separate question (drop it?) in its own log.

## Log

- 2026-10-11 01:51Z claude-ci@caspers-macbook: created
- 2026-10-11 01:51Z claude-ci@caspers-macbook: handoff -> blocked
