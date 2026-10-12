---
id: T0013
title: "P feature screen: p2 vs p3 at equal full-turn time, V off, and extraction cost"
status: review
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

*2026-10-12 06:27Z codex@desktop-g60ojud:*

PR29 https://github.com/Casper015/ctf-fusion/pull/29 open for review; head c3b6e59cd87b4b6d68b3f1e4a265dd135e9d4bea, branch codex/t0013-p-features-host-evidence. All1960 official games (seed20261210, Voff, 2000/50ms, sixworkers) and actual-weight cost probes complete. v12 P3-P2 +.01102 CI[-.01761,.03965], C08+.00209[-.02559,.02977]: both and every prespecified band/reference comparison/interaction not shown. AA all280scorevectors identical. Independent audit checks1960rawgames/784000turnblocks/6664000allseat replies; zeroallseatfault/missinginvalid/candidatelate. Core224=223pass1platformskip, layout6pass; report/source/hash audits retained. P3costhigher, C08samplesP2/P3 360.1/276.0; not enough strength evidence to change defaults. No replay/export/retraining/defaultchange. reports bots/p_only/report/p_features/RESULTS.md and reports/host/feature_cost_DESKTOP-G60OJUD.*; user outputs/T0013 contains21GitLF reports/DELIVERY/transport. CI pending on exacthead; keep review until accepted. Finishedstages neverrestart, consumedseedneverreuse. T0032 remains separate candidate/budget, offline teacher not deployed.

## Log

- 2026-10-10 19:24Z claude@caspers-macbook: created
- 2026-10-12 06:27Z codex@desktop-g60ojud: handoff -> review
