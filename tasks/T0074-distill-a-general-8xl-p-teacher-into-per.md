---
id: T0074
title: "Distill a general 8XL P teacher into per-count XXL students"
status: active
owner: claude-fervent@cloud
needs: [host, cuda, ai-test]
writes: [training/distill_count_specialists.py, reports/models/count_specialists_distill/, inference/count_specialists_distill/, profiles/count_specialists_distill.json, tests/test_repo_layout.py]
depends: [T0073]
land: pr
branch: claude-fervent/t0074-distill-a-general-8xl-p-teacher-into-per
updated: 2026-10-18
---

## Goal

Distill one general 8XL P teacher into per-count XXL students, to test whether a stronger teacher improves the T0067 per-count P bank. Offline training and test only; any game screen is a separate task.

Gate: this task waits on T0073 (agent_v2 with the T0067 bank vs the XXL P). Start only if T0073 shows the per-count bank helps in games. If T0073 is "not shown" or "hurts", hand off --status blocked and ask Casper whether to run it anyway; better offline P loss would then not be the bottleneck.

Evidence so far: T0032 (reports/models/p_distill/RESULTS.md): an 8XL p3 teacher trained on 24.15M rows beat XXL by 0.028 validation loss; distilling it into XXL students gained 0.0116 on all 3 seeds (42%), almost all in the joint head; offline only, never played. T0071 (reports/models/count_specialists_8xl/): per-count 8XL from random init on T0066 data (112k-840k train rows per count) is worse than XXL on all 14 counts, and still improving at epoch 8, so it measures init and budget, not data sufficiency. Do not reuse T0071 models or conclusions.

1. Teacher: one general p2 8XL (mover 2048, squad 8192-2048, act 2048, target 256; the T0032 widths) trained on the P2 corpus behind inference/models_xxl (data/shards on the host, about 24M rows) plus the T0066 train splits of all 14 counts. Never use T0066 val/test rows for training. Exclude rows of the held-out screen lineages (profiles/meow_rl_screen.json reference roster) if the shards identify them; otherwise report that the corpus contains them. Train to convergence (validation loss no longer improving), not a fixed 8 epochs, and record the schedule. Gate: the calibrated teacher must have lower T0066 val loss than both the deployed XXL and the T0067 specialist at most counts; otherwise stop and report.
2. Students: per count, start from the T0067 specialist (inference/count_specialists_p/weights_nXX.npz) and fine-tune on that count train split with fusion/policy_distill.py (alpha 0.5, tau 1, as T0032). Control arm: the same fine-tune with alpha 1 (no teacher), so extra training is separated from distillation. Seeds 101/102/103 for both arms, the same selection rule, temperatures fitted on val.
3. Export: int16 headers with their own namespace and guard (not csp_nXX, not nXX), MANIFEST and profile generated from the npz files with hashes, C++/Python parity on every header, and the coexistence check against models_xxl and count_specialists_p.
4. Test, once after selection: int16 student minus int16 control and minus the T0067 specialist, per count, 95% map-cluster intervals over the 30 test maps, Bonferroni over the 14 counts for per-count claims, plus the equal-weight mean over counts.
5. Report under reports/models/count_specialists_distill/: teacher curves and gate, per-seed student results, the tables above, and every number computed from receipts. No verdict labels without a stated rule.

## Done when

Teacher trained and gated (or stopped with the gate result); 14 students and 14 controls on 3 seeds; int16 export with parity and coexistence; test tables with map-cluster intervals and multiplicity; PR open. Or a blocked handoff if T0073 did not show the per-count bank helps.

## Handoff

*2026-10-16 20:10Z antigravity@desktop-g60ojud:*

T0073 paired carrier screen returned not shown (overall mean diff -0.0116 [-0.0315, +0.0082], 1,176 games). Per T0074 gate rule, better offline P loss is not the bottleneck in games, so 8XL distillation is blocked pending Casper's explicit decision.

## Log

- 2026-10-16 19:37Z claude-review@caspers-macbook: created
- 2026-10-16 20:10Z antigravity@desktop-g60ojud: handoff -> blocked
- 2026-10-17 04:07Z claude-review@caspers-macbook: note: Casper's decision (2026-10-17): drop T0074. Its gate is unmet (T0073 not shown, -0.0116 [-0.0315, +0.0082]). Owner: run task.py drop T0074 with that reason. Reopen P distillation only if a P consumer, such as P inside search (T0079 follow-up), shows a prediction bottleneck.
- 2026-10-18 19:38Z claude-fervent@cloud: note: Casper (2026-10-18, Claude session): close or drop this task; it has sat in review/blocked for days. Taking it with claim --force on that approval to change its status.
- 2026-10-18 19:38Z claude-fervent@cloud: claimed (forced)
