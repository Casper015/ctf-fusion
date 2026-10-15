---
id: T0013
title: "P feature screen: p2 vs p3 at equal full-turn time, V off, and extraction cost"
status: done
owner: codex@desktop-g60ojud
needs: [host]
writes: [bots/p_only/report/p_features/, reports/host/feature_cost_DESKTOP-G60OJUD.json, reports/host/feature_cost_DESKTOP-G60OJUD.md]
depends: [T0011]
land: direct
branch: ""
updated: 2026-10-12
---

## Goal

Games half of the P feature ablation (docs/FEATURE_FAMILIES.md section 8): the retrained p2 and p3 XXL policies of T0011 in both carriers, V off, fixed-time deployment, tuning seed 20261210, A/A copy, with the committed XXL P-only bots as references. On DESKTOP-G60OJUD after T0011 is merged (pull, then no pull or rebuild during runs): (1) extraction and inference cost with real weights: python tools/repo/task.py lock cpu --task T0013 -- python tools/host/value_cost.py --set inference/models_pf_p2xxl_vm --set inference/models_pf_p3xxl_vm --calls 3 --copies 1 6 15 --out reports/host/feature_cost_DESKTOP-G60OJUD (policy_view_us includes the p3 combat summary; policy_encode_us and policy_joint_us per team). (2) python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_p_features.json --dry-run, then python tools/repo/task.py lock cpu --task T0013 -- python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_p_features.json (6 arms + 1 A/A x 280 fixtures = 1,960 games). Report per band with 95% intervals, faults, late replies, p99 and the search counters (C08 samples per call show what p3 extraction costs the search). Intervals including zero are 'not shown'; read single bands against the A/A row.

## Done when

bots/p_only/report/p_features/ and reports/host/feature_cost_DESKTOP-G60OJUD.* committed; Handoff states the p3 - p2 verdict per carrier and band, the extraction cost, and whether p3 goes to confirmation (T0006).

## Handoff

*2026-10-12 06:50Z codex-t13-review@caspers-macbook:*

Completed: PR #29 https://github.com/Casper015/ctf-fusion/pull/29 reviewed and merged as 9bbe426b40401c1cdb23306c84330fbdc14e06ca; reviewed head c3b6e59cd87b4b6d68b3f1e4a265dd135e9d4bea. Casper explicitly authorized reviewer completion on 2026-10-11 (America/Los_Angeles). Independent delivery reviewer: codex-t13-review@caspers-macbook; original implementation owner retained.

Experiment delivered: 280 paired independent map fixtures / 1960 official 2000/50 ms games, V off, seed 20261210 consumed. v12 P3-P2 +0.01102 CI[-0.01761,+0.03965]; C08 +0.00209 CI[-0.02559,+0.02977]. Both carriers and all prespecified bands show no demonstrated strength gain. A/A score vectors identical on all 280 maps. Zero all-seat faults, missing/invalid replies and candidate late replies in the committed host audit. Actual-weight cost artifacts present; P3 costs more and C08 sample ratios remain descriptive.

Acceptance verified against the task: report-only scope, 1960 raw-receipt identities, 784000 turn-block / 6664000 reply arithmetic, published comparison/audit consistency, all five embedded source hashes/byte counts and Python syntax. Inspected independent audit score/utility/pairing/interval logic. Exact-head CI and Bots workflows succeeded. Reviewer did not rerun the retained host raw corpus, games, cost probes or predictions; SEARCH ratios and individual timing-percentile recomputation limits remain explicit.

Reports: bots/p_only/report/p_features/RESULTS.md and reports/host/feature_cost_DESKTOP-G60OJUD.*. Defaults/models stay unchanged. T0013 is complete as an experiment, not P3 deployment approval. T0032 is separate; do not repeat finished stages, add endpoint games or reuse seed 20261210.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-12 06:27Z codex@desktop-g60ojud: handoff -> review
- 2026-10-12 06:50Z codex-t13-review@caspers-macbook: user-authorized review accepted -> done; PR #29 merged 9bbe426b40401c1cdb23306c84330fbdc14e06ca; implementation owner retained