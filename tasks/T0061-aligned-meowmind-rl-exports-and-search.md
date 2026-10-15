---
id: T0061
title: "Aligned Meowmind RL exports and search diagnosis: shared calibration corpus, WideAct16/FP32 critics, search v2, fixed trees, eight-worker gate (code)"
status: active
owner: claude-review@caspers-macbook
needs: []
writes: [fusion/quant_wide.py, inference/mlp_wide.hpp, inference/mlp_float.hpp, bots/meow_rl/, tools/cpp/meow_rl/tree.cpp, tools/cpp/meow_rl/wide_check.cpp, tools/meow_rl_aligned/, bots/p_only/tools/preflight_meow_rl.py, profiles/meow_rl_aligned_screen.json, docs/MEOWMIND_RL_ALIGNED.md, docs/MARKDOWN_INDEX.md, tests/test_meow_rl_aligned.py]
depends: []
land: pr
branch: claude-review/t0061-aligned-meowmind-rl-exports-and-search
updated: 2026-10-15
---

## Goal

Casper asked (2026-10-15) for an aligned evaluation of the Meowmind RL actors and a diagnosis of why value search loses (T0057: search - L = -0.148). This task is the code; the host tasks that depend on it run it. No new training.

1. Shared calibration corpus. A deterministic, stratified selector of 16,000 learner decisions from T0059's retained batches (runs/meow_rl_pool2/large_v_s101/batches, update_*.npz with feats and value_feats). The strata are team count 2-15, early/middle/late turn, carrier, combat, low HP and respawn, where the features show them. It also writes a disjoint frozen diagnostic split. Its manifest records parent-file hashes, (episode, turn, row) indices, the selection seed and rules, and the missing strata.
2. Exports of all four actors against that corpus with one configuration: the warm start, T0055 large_v, T0059 large_v and T0059 small_v. Each gets an experiment-specific name and never reuses a historical one. Checks:
   - the same inputs give the same digest;
   - the warm start through the reference and array paths gives identical tensors;
   - greedy agreement, TV/KL, saturation, and probability parity including masks, T and the sampling draw.
3. Critic backends for the T0059 large-V checkpoint's critic, all with the same reference16 actor header:
   - F: the checkpoint's FP32 critic on CPU;
   - R: the reference16 critic;
   - H: WideAct16, i.e. INT16 weights, 15-bit hidden activations and INT64 accumulators. Requirements: a formal per-layer bound proof, defined rounding without implementation-defined shifts, Python emulation, native scalar and SIMD paths, and parity tests.
4. A versioned search (MEOW_RL_SEARCH 2) that leaves the T0056/T0057 search unchanged. It uses common random numbers across root candidates, and an admission budget that checks time before ply 1, before the ply-2 expansion, before the critic and after the evaluation, with a ready actor fallback. It also emits work counters.
5. A fixed-tree harness: one materialized, hashed search tree per diagnostic root, whose leaves are scored by F, R and H. It reports value MAE/RMSE/max/bias, within-root ranking disagreement, top-action agreement and margins.
6. The eight-worker preflight (counts 2, 5, 8, 15, 16 fixtures each per variant) with first and steady timing separate. Plus the frozen screen specifications A (aligned actor-direct, 392 fixtures x 5 arms) and B (D/F/R/H + A/A of R, 392 x 5), with Bonferroni (2 and 5 tests, computed quantile).
Spec: docs/MEOWMIND_RL_ALIGNED.md (written by this task).

## Done when

PR merged: tests cover the selector, export reproducibility and parity, the WideAct16 bounds (Python and native scalar/SIMD), FP32 critic parity, search v2 equivalence and deadline behaviour, the tree harness and the eight-worker preflight. docs/MEOWMIND_RL_ALIGNED.md gives the exact host commands for the dependent host tasks.

## Handoff

Not started.

## Log

- 2026-10-15 19:58Z claude-review@caspers-macbook: created
- 2026-10-15 19:59Z claude-review@caspers-macbook: claimed
- 2026-10-15 19:59Z claude-review@caspers-macbook: amended goal; done when
