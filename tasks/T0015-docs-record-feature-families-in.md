---
id: T0015
title: "Docs: record feature families in CONTRIBUTING, README and the P-only experiment notes"
status: dropped
owner: ""
needs: []
writes: [CONTRIBUTING.md, README.md, bots/p_only/EXPERIMENTS.md]
depends: [T0009]
land: direct
branch: ""
updated: 2026-10-11
---

## Goal

T0009 could not touch files outside its lanes. Add: in CONTRIBUTING.md, generated-file rows for bots/p_only/include/ponly_p3.hpp, bots/p_only/bots/v12_ponly_f.hpp and c08_ponly_f.hpp (bots/p_only/tools/derive/make_families.py; bots/p_only/tests/test_families.py re-derives them), the run-based mode of training/export_mixed.py (--policy-run, --value-run; shared V header), and reports/models/v_pilot/ and p_features/ as study folders (tests/test_repo_layout.py MODEL_STUDIES). In README.md, a line for docs/FEATURE_FAMILIES.md. In bots/p_only/EXPERIMENTS.md, a pointer to the family variants (profiles/variants_families.json) and screens screen_p_features.json / screen_v_pilot.json / collect_v_data.json with their seeds (20261201 data only, 20261210, 20261211).

## Done when

The three files point to docs/FEATURE_FAMILIES.md and the new generators; tests pass.

## Handoff

*2026-10-11 01:32Z claude-ci@caspers-macbook:*

Merged into T0024 (Board and docs consolidation: merge duplicate tasks, task summary, markdown digest and index) by claude-ci@caspers-macbook: the same documentation work: T0015's three doc edits and T0008's triage of the pre-board prompts are part of reading and summarising every markdown file

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-11 01:32Z claude-ci@caspers-macbook: merged into T0024
