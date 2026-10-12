---
id: T0033
title: "RL plan: preference, offline RL and expert iteration on top of P (design doc)"
status: active
owner: claude-ci@caspers-macbook
needs: []
writes: [docs/RL_PLAN.md, docs/MARKDOWN_INDEX.md, docs/DIGEST.md]
depends: []
land: direct
branch: claude-ci/t0033-rl-plan-preference-offline-rl-and-expert
updated: 2026-10-12
---

## Goal

Write docs/RL_PLAN.md: a staged, cost-ordered plan to replace or augment supervised V with an RL-trained own-move component (P predicts opponents, RL moves us). Stage 1 (preference optimization on T0010's branch pairs) in implementable detail with commands, files, report paths and numeric kill/pass rules; stages 2 (offline RL) and 3 (expert iteration) sketched and costed from measured host wall times. Flag every repository fact that contradicts the request's premises. No code, training, games or seeds spent; proposed follow-on tasks are listed, not created.

## Done when

docs/RL_PLAN.md landed on main and indexed (MARKDOWN_INDEX, DIGEST pointer); the Handoff lists the proposed tasks and the open decisions for Casper.

## Handoff

Not started.

## Log

- 2026-10-12 04:52Z claude-ci@caspers-macbook: created
- 2026-10-12 04:52Z claude-ci@caspers-macbook: claimed
- 2026-10-12 04:59Z claude-ci@caspers-macbook: landed: 753c624ee2 on main, 3 file(s)
