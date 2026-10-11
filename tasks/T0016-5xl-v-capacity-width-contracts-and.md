---
id: T0016
title: "5XL V capacity: width contracts and reproducible experiment"
status: review
owner: codex@desktop-g60ojud
needs: []
writes: [fusion/, training/, inference/mlp.hpp, inference/value_v3.hpp, tests/, tools/cpp/, tools/host/value_cost.py, profiles/v_capacity_5xl.json, docs/V_CAPACITY_5XL.md, reports/models/v_capacity_5xl/]
depends: []
land: pr
branch: codex/t0016-5xl-v-capacity-width-contracts-and
updated: 2026-10-10
---

## Goal

Implement and validate explicit v3 XL 1024,64; 4XL 8192,512; 5XL 16384,1024 capacity comparison. Preserve production and T0012 four-arm scope. Audit native buffers, safe quantization, deterministic chunked export, bounded GPU profiles, offline paired decision metrics and hand off host-dependent evidence.

## Done when

PR with passing width/export/parity contracts, reproducible same-data FP32 pilot commands, parameter/MAC counts, capability and evidence status; real training/host evidence where available without changing production.

## Handoff

*2026-10-10 23:38Z codex@desktop-g60ojud:*

PR: https://github.com/Casper015/ctf-fusion/pull/11 (open, clean merge against current main). Branch codex/t0016-5xl-v-capacity-width-contracts-and; latest commit 32ab0de. Core implementation/training snapshot 130f766; measured host snapshot 4cbf07b. Production models and bot defaults were not changed by this task. T0012's four-arm scope was not changed.

Worktree retained on DESKTOP-G60OJUD:
C:\Users\caspe\Documents\Codex\2026-10-10\files-pasted-by-the-user-work\work\ctf-fusion.worktrees\T0016
Do not remove it: ignored runs/data/build contain the actual checkpoints, caches, exports, binaries and raw evidence.
CUDA Python: C:\Users\caspe\AppData\Local\Python\pythoncore-3.14-64\python.exe (Torch 2.11.0+cu128, RTX 4080 Laptop).
AI_TEST_ROOT: C:\Users\caspe\OneDrive\Code\AI_Test.
CTF_AGENT=codex@desktop-g60ojud; CTF_CAN=host,cuda,ai-test.

Implemented: explicit v3 108->16384->1024->3 (18,567,171 training parameters; 18,547,712 outcome MACs), deterministic fragmented export, one shared V and exact eleven frozen P arrays, transitive native dependencies, small-P scratch dispatch, no redundant v3 accumulator clearing, bounded GPU windows/validation/calibration, complete memory/effort records, paired root regret and pair baselines, bounded teacher targets, optional complete-order branch alternatives, concurrent-feed host benchmark, conservative identity-bound tiny-screen summary.

Trained from fresh initialization, seed 1, same 390,216 train / 52,728 val rows, source weights and map splits: XL, 4XL, 5XL, Student. Target is the original eight-binary C08/V12 mixed-policy designated dataset, not one frozen C08. Real float model_best.npz and calibrated model.npz stay in runs/v_capacity_5xl/{XL,4XL,5XL,Student}_s1/. 5XL best checkpoint is 74,270,122 bytes. External committed V3-XL reference was retained and a fresh XL control was retrained. No original final-test model predictions were evaluated.

Evidence in reports/models/v_capacity_5xl/SUMMARY.md, SUMMARY.json, VERIFICATION.json, source_audit.json, validation.json/md, sweeps, exact training/host argv, integer checks and run-config/curve snapshots. 5XL-XL validation log loss -0.001579, interval [-0.002716,-0.000443], 31 maps. Paired candidate regret improvement NOT SHOWN (8 maps). Trained native full/incremental row-major/transposed checks: 256 validation rows each; Python parity 16; recorded streams also exact. 5XL max probability quantization error 0.001104; layer bounds remain within +/-2^30. Shared V header/fragments total 94,280,572 bytes, 42 fragments, largest 2,809,540 bytes; generated exports/bundles are ignored under build/.

CPU-locked equal-time and fixed-work screens: 64 games, only four independent maps shared across modes, official 2s/50ms, four workers, zero candidate faults/late. Direct 5XL max reply 31.059ms time / 24.580ms work. All observed arms meet these limits; this is not all-count/load confirmation. V-only p99 full cost 1.442ms single process / 2.909ms fifteen concurrent processes. Counter lines missing on the two-team match are not treated as zeros; rates cover the three logged games. Time mode frequently exhausts the V deadline before reaching V and falls back. A/A variation is large. Decision and game-strength improvement NOT SHOWN; default remains unchanged.

Retained complete-order collection: data/branches/v_capacity_5xl_complete_isolated, frozen cap_xl_time binary (C08, p2 XXL, seed-1 XL V, max_calls=8/deadline=20), same opponent identities and exploratory maps. Three successful independent roots, one dropped, six candidates x two real continuations = 36; zero faults/prefix faults, prefix mismatch min1/max12/mean5.25. Map-derived branch splits retained. Ranking labels at data/rank/v_capacity_5xl_complete_isolated: train one resolved pair, val/test zero. No model scoring on its test split. An earlier plumbing collection is not evidence. These few noisy roots do not select a model.

Checks: full repository suite at af18d5c completed 160 tests, one skip, success; 20 P-only contracts passed; eight report provenance/layout follow-up tests passed; thirteen capacity-profile/layout checks passed after marking the seed consumed. The first final-suite run found the missing study registration; it was fixed and the entire suite rerun successfully. Logs remain under build/; hashes are in VERIFICATION.json. Git status clean. All new tracked files are below the repository size limit (the push's large-file warning refers to an unchanged pre-existing roster/report/results/behavior_distance.json blob).

Seed ledger: 20261220 was reserved by T0016 before T0019 and consumed only as exploration, never confirmation. T0019 subsequently specified the same seed as fresh; a task.py note now warns it to choose/reserve a different unseen seed before host execution. Previously used 20261031/20261102/20261107/20261108 and other tasks' reservations 20261201/20261210/20261211 cannot be called fresh. Leave the confirmation seed unchosen here.

Current main advanced during the frozen evidence runs: PR #10 adds default-off search portfolio/ply-2 changes. No pull or rebuild occurred during measurements. PR #11 merges cleanly with it, but the recorded host evidence is pinned to 4cbf07b and its carrier/source hashes; do not relabel it as measurement of a newly rebuilt carrier.

Review/continuation from the retained T0016 worktree, using the verified Python as `python`:

```powershell
python -m training.summarize_value_capacity --build build/v_capacity_5xl --out reports/models/v_capacity_5xl
python -m training.check_value_capacity --set build/v_capacity_5xl/sets/5XL --run runs/v_capacity_5xl/5XL_s1 --cache data/vcache/designated_v3 --out reports/models/v_capacity_5xl/integer_5XL.json
python -m training.evaluate_value --cache data/vcache/designated_v3 --split val --batch 2048 --run Reference=runs/reference_v3xl --run XL=runs/v_capacity_5xl/XL_s1 --run 4XL=runs/v_capacity_5xl/4XL_s1 --run 5XL=runs/v_capacity_5xl/5XL_s1 --run Student=runs/v_capacity_5xl/Student_s1 --rank data/rank/c08_v3xl --compare 4XL:XL --compare 5XL:XL --compare 5XL:4XL --compare 5XL:Reference --compare Student:XL --compare Student:5XL --out reports/models/v_capacity_5xl/validation
python tools/repo/task.py lock --task T0016 cpu -- python tools/repo/task.py lock --task T0016 gpu -- python -m unittest discover -s tests -t .
```

For exact numerical regeneration, preserve the measured source/binary snapshot; record any latest-main compatibility work separately. Existing screen analyzer results stay ignored in build/v_capacity_5xl/screen_{time,work}_analysis: one map/count cannot estimate its within-count intervals. Use SUMMARY's paired four-map t intervals and A/A diagnostics instead. Profiles/commands are in profiles/v_capacity_5xl.json, docs/V_CAPACITY_5XL.md and *_commands.json. Put --task BEFORE the positional cpu/gpu name because current task.py argparse captures the remainder.

Further deployment work is deliberately gated: expand a specified frozen C08 continuation training target with bounded complete orders and more repeated roots/maps; train/fine-tune all compared capacities on identical new rows; use validation only for selection/calibration and budget tuning; repeat selected finalists with independent seeds 1,2,3; freeze binaries/settings; then reserve a new confirmation seed and run the repository's 10,000 independent map groups with >=500/count 2-15 and required opponent holdouts. Do not run that confirmation or replace production on the basis of this tiny one-seed mixed-policy pilot. Follow-up collection/confirmation needs its own scoped task, not expansion of T0012. T0016 is now awaiting code/evidence review, not claiming confirmed strength.

## Log

- 2026-10-10 22:37Z codex@desktop-g60ojud: created
- 2026-10-10 22:37Z codex@desktop-g60ojud: claimed
- 2026-10-10 22:45Z codex@desktop-g60ojud: note: Confirmed host,cuda,ai-test via supplied Python 3.14 / torch 2.11.0+cu128 and AI_Test path. Actual 5XL shape and native integer/incremental/chunk contracts pass. Separate mixed-policy exploratory pilot uses cache ccf6f46945b83ad3 with fresh XL/4XL/5XL controls; production and T0012 untouched.
- 2026-10-10 22:56Z codex@desktop-g60ojud: note: Implementation frozen at 130f766. Repository 155 tests passed (1 skip), final CUDA/AI_Test contracts 53 passed, P-only 20 passed. Reserve 20261220 for this bounded time/work paired exploratory screen, not confirmation. Starting CPU+GPU locked sweeps and fresh capacity controls; no rebuild or pull during evidence.
- 2026-10-10 23:38Z codex@desktop-g60ojud: handoff -> review
- 2026-10-10 23:38Z codex@desktop-g60ojud: note: PR https://github.com/Casper015/ctf-fusion/pull/11 at 32ab0de is ready for review; real one-seed XL/4XL/5XL/student checkpoints and 64 official-limit host games retained. Full suite 160 tests/1 skip passed; no decision/strength gain shown; production unchanged. Detailed Handoff records commands and T0019 seed collision.
- 2026-10-11 01:13Z claude@caspers-macbook: note: The owner (human) asked claude@caspers-macbook to close T0016 after review; taking it with claim --force for that only. PR #11 merged as 72a3740 (merged with main locally: 165 core tests OK, 13 skipped without torch; P-only 43 OK). Review follow-ups landed on main in 787e535: docs/V_CAPACITY_5XL.md names the evidence commit 4cbf07b and carrier hash, records the low V use in the equal-time screen, notes the pair-accuracy tie change, updates the seed ledger (T0019 20261230, T0022 20261240); value_v3.hpp comment corrected. Handoff above kept unchanged; the host worktree and ignored runs/data/build stay on DESKTOP-G60OJUD.
