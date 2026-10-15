---
id: T0038
title: "Audit expanded decision pilot and build source-bound training dataset"
status: done
owner: codex-t38-review@desktop-g60ojud
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

*2026-10-15 01:50Z codex-t38-review@desktop-g60ojud:*

Completed and independently accepted. Implementation by codex-data-audit@desktop-g60ojud; final review and supplemental acceptance by codex-t38-review@desktop-g60ojud. PR https://github.com/Casper015/ctf-fusion/pull/46 merged as 7cd75b2b0d07d49ac8cca458cfe5fb9ae798f1ab; exact head 711a73addfce8dc18356ac23a7f7fc05f229fe21 passed all nine verification CI jobs (release skipped as expected).

Deliverables: bots/p_only/report/decision_dataset/ and reports/eval/decision_data_quality/. DATA_AUDIT.json, FINAL_ACCEPTANCE.json, CHECKS.json, the executed read-only acceptance source, precise-summary generator, and REPRODUCE.md preserve the audit and reproduction evidence. User copies and exact-head CI are in outputs/T0038/ under the host work root.

Accepted the exact completed T0037 corpus: plan 8a4a0417ae7d765cf54a2b9b705b0bd79a6e3bd36b1ba18c66876c1e223cadb5, seed 20261701 already consumed, 336 maps/roots, 1317 candidates, 5268 continuations, zero drops and zero all-phase faults. Final independent checks bind 53 actual frozen binaries, engine, 2107200 saved trajectory blocks, per-repeat raw scores and input protocol, canonical JSON prefix bindings, and all full-prefix native NPZ features/canonical actions. All 18972 normalized prefix order mismatches remain as replay-memory diagnostics.

Existing authoritative ctf-fusion-decision-2 dataset: data/decision/host_pilot_dataset/{train,val,test}/decision.npz with 268/37/31 maps and 1048/148/121 candidates. SHA256 train 70441e0197382c630b2f64f230e54722ae031bb184e5a73e3874e08b00b877bc; val 74b630efa4e80bd2f947c45a816751124b06c155bc6e05533d88adb0255b5ace; test 1b9a0dd6719c27d98e621248dda06c841df0b54a61ea3edb5f8d5d8541cce598. Final review did not rebuild this dataset or replay any successful game/continuation.

Precise equal-count independent-repeat headroom: train +0.05252749281939906 [0.03447802561563207, 0.07057696002316605]; validation +0.08953179299333147 [0.01763937086498979, 0.16142421512167315]. The summary precision correction changes no raw/NPZ bytes or recommendation. GO for a later small linear/MLP own-action trial, not an admission of a trained policy or strength gain. Retain selected-frozen-identity scope (original full-manifest baseline unavailable), flag-selector fallback/limited requested coverage, replay-memory effects and only 31 independent test maps. No test-outcome tuning, training, deployment, model changes or production-default changes started.

Local publication checks: initial core 324 tests had 300 pass/23 skip and one archived-link failure resolved by incorporating the already merged main fix; subsequent layout/board suite 27 passed. Local Windows P-only 56 tests had 52 pass/3 skip and one unchanged test_decision_scoring.py:216 filename assertion: g++ successfully creates .exe while the test checks an unsuffixed path. A minimal compiler filename probe confirms the behavior; raw logs are preserved. Final exact-head CI, including P-only and Windows core, passed. No extra code lanes were edited to mask this platform assertion. No follow-up training or old queue is automatically resumed.

## Log

- 2026-10-12 08:51Z codex-data-planning@caspers-macbook: created
- 2026-10-12 20:26Z codex-data-audit@caspers-macbook: note: Independent pre-audit requested by Casper, snapshot 3f88bbbb: T0038 has not started and completed branch backup/dataset receipts are not yet available. Rehashed all631 stage-1 payload files and checked archive SHA2d046297; existing decision-learning contract tests35 OK (toy/recorded-input tests only). Synthetic diagnostic counterexample: adding an unsupported root leaves usable between-action raw variance0.18 unchanged but shifts repeat_stats net0.17975->0.148625, because its noise pool includes a different population. Before a go/no-go decision, report root-matched correction and paired-gap variance/covariance limits. cand_faults counts live continuation faults only: independently retain/aggregate prefix, discarded computed root and continuation phase diagnostics. Verify exact candidate/repeat IDs, effective support, original manifest/binary identities and source-bound per-split receipts once the completed backup arrives. No passed audit receipt/dataset/training admission; no successful games rerun, real training started or running Colab job modified. Keep this task open and dependent on T0037 completion.
- 2026-10-14 19:24Z codex-host-handoff@caspers-macbook: note: Casper moved T0037 back to Windows desktop-g60ojud on October14. Audit its completed new host corpus once T0037 is done; keep Colab history and any auxiliary analysis separate. Preserve the recorded pre-audit noise-population and all-phase-fault checks. No dataset authority or training admission before exact source-bound audit.
- 2026-10-14 23:57Z codex-data-audit@desktop-g60ojud: claimed
- 2026-10-15 00:34Z codex-data-audit@desktop-g60ojud: handoff -> review
- 2026-10-15 01:21Z codex-t38-review@desktop-g60ojud: claimed (forced)
- 2026-10-15 01:26Z codex-t38-review@desktop-g60ojud: note: Final independent raw/NPZ acceptance passed: 53 actual binaries, 336 roots, 5268 continuations, 2107200 stored trajectory blocks, all-phase faults zero. Publication-only precision refresh regenerates summary receipts from existing hash-pinned train/val NPZs; detailed noise and recommendations unchanged. No successful games replayed, dataset rebuilt, training or deployment started. Final review fix and exact-head CI pending on PR46.
- 2026-10-15 01:50Z codex-t38-review@desktop-g60ojud: done
