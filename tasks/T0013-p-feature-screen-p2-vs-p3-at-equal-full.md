---
id: T0013
title: "P feature screen: p2 vs p3 at equal full-turn time, V off, and extraction cost"
status: open
owner: ""
needs: [host]
writes: [bots/p_only/report/p_features/, reports/host/feature_cost_DESKTOP-G60OJUD.json, reports/host/feature_cost_DESKTOP-G60OJUD.md]
depends: [T0011]
land: direct
branch: ""
updated: 2026-10-10
---

## Goal

Games half of the P feature ablation (docs/FEATURE_FAMILIES.md section 8): the retrained p2 and p3 XXL policies of T0011 in both carriers, V off, fixed-time deployment, tuning seed 20261210, A/A copy, with the committed XXL P-only bots as references. On DESKTOP-G60OJUD after T0011 is merged (pull, then no pull or rebuild during runs): (1) extraction and inference cost with real weights: python tools/repo/task.py lock cpu --task T0013 -- python tools/host/value_cost.py --set inference/models_pf_p2xxl_vm --set inference/models_pf_p3xxl_vm --calls 3 --copies 1 6 15 --out reports/host/feature_cost_DESKTOP-G60OJUD (policy_view_us includes the p3 combat summary; policy_encode_us and policy_joint_us per team). (2) python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_p_features.json --dry-run, then python tools/repo/task.py lock cpu --task T0013 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_p_features.json (6 arms + 1 A/A x 280 fixtures = 1,960 games). Report per band with 95% intervals, faults, late replies, p99 and the search counters (C08 samples per call show what p3 extraction costs the search). Intervals including zero are 'not shown'; read single bands against the A/A row.

## Done when

bots/p_only/report/p_features/ and reports/host/feature_cost_DESKTOP-G60OJUD.* committed; Handoff states the p3 - p2 verdict per carrier and band, the extraction cost, and whether p3 goes to confirmation (T0006).

## Handoff

Not started.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
