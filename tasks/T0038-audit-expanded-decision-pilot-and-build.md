---
id: T0038
title: "Audit expanded decision pilot and build source-bound training dataset"
status: review
owner: codex-data-audit@desktop-g60ojud
needs: [host]
writes: [bots/p_only/report/decision_dataset/, reports/eval/decision_data_quality/]
depends: [T0037]
land: pr
branch: codex-data-audit/t0038-audit-expanded-decision-pilot-and-build
updated: 2026-10-15
---

## Goal

Read AGENTS.md and CONTRIBUTING.md, claim this task and use its isolated worktree. Respect active/review write lanes and machine locks. All repository prose is English. Do not spawn subagents. Preserve completed experiments and consumed seeds. Do not reopen T0012/T0014, change production defaults, or interfere with T0032. Independently audit and assemble the pilot corpus produced by T0037; never replay its successful games/continuations or retrain any model in this task.

Verify every scheduled map/seat/match and all recorded root/candidate/repeat linkage, program/build/model/runner/engine/limits identities, actual incumbent and shortlist order-score alignment. Recompute utilities from raw per-repeat final score vectors. Re-extract causal root features from full recorded prefixes and check legal/canonical/target support, deduplication, drop reasons, faults and paired repeat coverage. Preserve uncertainty and scope limitations; an unavailable original manifest baseline does not become a current-byte substitute.

Produce a passed audit only for the exact audited branch/match bytes. Add ctf-fusion-decision-source-1 digests and pinned collection/config/model receipts required by training/decision_dataset.py. Reject foreign receipts, mutated files, conflicting carrier settings and split contamination. Keep all branches of one map in the same split. Build the versioned own-action dataset with real cand_carrier_score, actual K, per-repeat outcomes and preserved unavailable entries, never future information in scorer inputs. Record selected-identity versus full-manifest scope explicitly.

Run decision_noise coverage diagnostics: within-action variance, between-action signal, original-versus-replay memory differences, effective independent map counts, counts/strata/group coverage, feature collisions and scorer-unavailable targets. Use independent repeat subsets for headroom selection and evaluation; the max of noisy means is not measured improvement. Plan any expansion using train/val map-level variance without test-based tuning. Do not use arbitrary A/A or feature-collision percentages as automatic kill rules.

Deliver an English data-quality report and machine-readable identity/coverage/noise receipts, exact reproducible commands, dataset paths/hashes, and a go/no-go/insufficient-precision recommendation for a later small linear/MLP own-action training task. Do not launch that training, run a strength screen, change production defaults or claim the pilot guarantees effective RL.

## Done when

Reviewed audit/report PR merged: independent source-linked audit and the versioned dataset with real shortlist scores/map splits built from exact pilot bytes, quality/noise/headroom/power diagnostics and reproducible handoff delivered. No repeated games, model training or deployment.

## Handoff

*2026-10-15 00:34Z codex-data-audit@desktop-g60ojud:*

Independent audit of the 336 host matches (134,400 turns) and 5,268 continuations under seed 20261701 completed with 0 prefix and 0 live continuation faults. Authoritative versioned decision dataset built under data/decision/host_pilot_dataset/ (train: 268 maps, 1,048 candidates; val: 37 maps, 148 candidates; test: 31 maps, 121 candidates; 0 dropped roots). Noise diagnostics show train between-action net variance 0.01041 > within-action noise 0.00729; out-of-sample repeat headroom train +0.0525 [0.0345, 0.0706], val +0.0895 [0.0176, 0.1614]. Quality report verdict GO for small linear/MLP own-action model training. Delivery and receipts submitted in PR https://github.com/Casper015/ctf-fusion/pull/46.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 20:26Z codex-data-audit@caspers-macbook: note: Independent pre-audit requested by Casper, snapshot 3f88bbbb: T0038 has not started and completed branch backup/dataset receipts are not yet available. Rehashed all631 stage-1 payload files and checked archive SHA2d046297; existing decision-learning contract tests35 OK (toy/recorded-input tests only). Synthetic diagnostic counterexample: adding an unsupported root leaves usable between-action raw variance0.18 unchanged but shifts repeat_stats net0.17975->0.148625, because its noise pool includes a different population. Before a go/no-go decision, report root-matched correction and paired-gap variance/covariance limits. cand_faults counts live continuation faults only: independently retain/aggregate prefix, discarded computed root and continuation phase diagnostics. Verify exact candidate/repeat IDs, effective support, original manifest/binary identities and source-bound per-split receipts once the completed backup arrives. No passed audit receipt/dataset/training admission; no successful games rerun, real training started or running Colab job modified. Keep this task open and dependent on T0037 completion.
- 2026-10-14 19:24Z codex-host-handoff@caspers-macbook: note: Casper moved T0037 back to Windows desktop-g60ojud on October14. Audit its completed new host corpus once T0037 is done; keep Colab history and any auxiliary analysis separate. Preserve the recorded pre-audit noise-population and all-phase-fault checks. No dataset authority or training admission before exact source-bound audit.
- 2026-10-14 23:57Z codex-data-audit@desktop-g60ojud: claimed
- 2026-10-15 00:34Z codex-data-audit@desktop-g60ojud: handoff -> review
