---
id: T0033
title: "RL plan: preference, offline RL and expert iteration on top of P (design doc)"
status: done
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

*2026-10-12 04:59Z claude-ci@caspers-macbook:*

Landed 753c624: docs/RL_PLAN.md (plan only; no code, training, games or seeds spent), DIGEST entries for RL_PLAN and P_MIXED_PRECISION, index regenerated. Proposed tasks A-G in RL_PLAN section 7 are NOT created: A stage-1 code (branch_noise, pref_gate, dpo_policy, branch_dataset --root-records, own_pref export refusal); B stage-1 offline run on T0010 data (needs T0012 and the T0010 host re-audit, DATA_AUDIT source_row_alignment_verified=false); C carrier hook PONLY_PREF/PONLY_ROOT_LOG/PONLY_EXPLORE_EPS (after T0030); D P3 refresh collection seed 20261401; E 1-S screen seeds 20261411/20261410; F,G stages 2-3 after E. Decisions for Casper (section 8): RL carrier (v12_p3 recommended), meaning of three training seeds for expert iteration, Refresh rule text covering P/actor changes, recording T0010/T0011 as done on the board.

## Log

- 2026-10-12 04:52Z claude-ci@caspers-macbook: created
- 2026-10-12 04:52Z claude-ci@caspers-macbook: claimed
- 2026-10-12 04:59Z claude-ci@caspers-macbook: landed: 753c624ee2 on main, 3 file(s)
- 2026-10-12 04:59Z claude-ci@caspers-macbook: done