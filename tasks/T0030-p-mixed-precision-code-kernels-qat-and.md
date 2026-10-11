---
id: T0030
title: "P mixed-precision code: kernels, QAT and search interfaces"
status: review
owner: claude-ci@caspers-macbook
needs: []
writes: [CONTRIBUTING.md, bots/p_only/bots/, bots/p_only/include/, bots/p_only/profiles/, bots/p_only/tests/, bots/p_only/tools/derive/, docs/P_MIXED_PRECISION.md, fusion/, inference/mlp.hpp, inference/policy.hpp, inference/policy_p3.hpp, inference/policy_precision/, inference/simd.hpp, profiles/p_precision_code.json, tests/, tools/cpp/, training/]
depends: [T0009, T0018, T0020]
land: pr
branch: claude-ci/t0030-p-mixed-precision-code-kernels-qat-and
updated: 2026-10-11
---

## Goal

Design and implement experimental mixed-precision opponent-policy P infrastructure for CPU deployment. Deliver usable code and contracts now, while T0011 continues its frozen P2/P3 training. Preserve the existing INT16 inference path and all production defaults. This task prepares later teacher/student training and host evaluation; it does not run either.

Worker setup and scope

- Read AGENTS.md, CONTRIBUTING.md, docs/FEATURE_FAMILIES.md, docs/SEARCH_PORTFOLIO.md, this task, and the current T0011 handoff. Claim this task with a unique, truthful agent identity and work in its own worktree. It needs neither host nor CUDA capability for its code deliverables.
- Do not enter, modify, restart, pull, rebase or rebuild T0011's worktree or its running pipeline. Do not alter its memory guard, locks, inputs, checkpoints, calibration, seeds or exports. Source changes here belong to the new isolated worktree and a PR.
- Freeze features, game rules, P/V roles, search depth, candidate generation, evaluation coefficients and budgets for every controlled comparison. P predicts opponents; it is not a policy for our own units. V compression is outside this task.
- Implement and document in English. Use task.py notes/handoff for progress; no separate next-agent prompt files. Never commit raw datasets, checkpoints, generated experimental bundles or binaries.

1. Architecture and compatibility

Inspect the actual p2/p3 encoders, four policy heads, training backend, exporter, integer MLP/SIMD paths and family-carrier generators before choosing interfaces. Record the design in docs/P_MIXED_PRECISION.md with concrete file/module responsibilities and data flow. Add explicit quantization/profile identity and format metadata without weakening existing family, shape, buffer, overflow or provenance contracts. The legacy format and default-disabled carriers must retain their existing numerical behavior.

Provide three experimental precision profiles: reference16, uniform8, and mixed4_8. Their shapes and features match; quantify actual parameter/storage counts per head from arrays rather than tier names. Report packed weights, bias/scales, alignment/padding, scratch and resident copies separately. All activation ranges and rounding are explicit; the existing hidden range 0..128 fits uint8, not int8. Inputs may need the existing signed higher-precision path. Keep integer dot-product accumulators INT32 and prove intermediate bounds, including SIMD saturation and requantization. Probability normalization/calibration may retain the existing floating-point path.

2. IQ-inspired calibration and mixed precision

Borrow activation-aware error weighting, layer sensitivity and blockwise scaling from IQ/AWQ ideas. Collect bounded activation second-moment statistics on declared calibration rows and on recorded/synthetic successor states. Define weighted reconstruction error and measure changes in masked joint, action and target distributions. Protect important channels in contiguous hardware-friendly blocks or layers; avoid an unbounded sparse high-precision exception path.

Initial mixed candidate: low-bit weights in the large squad hidden matrix, higher precision in sensitive input/output layers and small action/target heads. Validate sensitivity instead of declaring any layer inherently critical. All calibration fits are training/validation-only; do not fit scales, routing thresholds or temperatures on a held-out test or confirmation seed. Rare carrier, low-HP, pickup/drop, delivery and combat contexts need explicit coverage diagnostics.

Start with packed uniform INT4/INT8. Literal IQ3 codebooks are not required for completion: document where an IQ3-style codec could fit, why its block/padding/decode overhead matters for these small MLPs, and how it would need separate quality and CPU measurements. A three-bit file size is not a speed guarantee.

Primary references: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-quants.c ; https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/imatrix.cpp ; https://github.com/ggml-org/llama.cpp/blob/master/src/llama-quant.cpp ; https://arxiv.org/abs/2306.00978 . Record the implementation revisions actually inspected.

3. Export and CPU execution

Implement deterministic packing, scale/bias metadata, manifest hashing, Python integer emulation and independent portable scalar reference kernels. Implement an optimized CPU path appropriate to the supported ISA, with runtime dispatch and a correct fallback; cover x86/ARM compilation and correctness without claiming timing on an unmeasured ISA. Packed weights must be consumed by the kernel through fused/block decode or integer operations, not permanently expanded into a full INT16 copy. Retain a common manifest identity across exported arrays, headers, shared chunks, bundled source and native build cache. Keep experimental exports under ignored build/ until a later model task approves them.

Fake quantization during training must reproduce deployment scales, clamps, rounding and requantization. INT4/INT8 parity means matching the reference for that profile, not being identical to INT16. Disabled mode must remain identical to the existing INT16 path. Prepare a reproducible benchmark command with CPU/compiler/flags/concurrency/checkpoint/profile identities, startup and warm full-reply measurements, and actual activation/weight traffic statistics where available. Do not execute host timing or game batches in this task.

4. Teacher/student and QAT code, without real training

Prepare a bounded-memory P distillation/QAT entry point reusing the existing policy training semantics. A future teacher-only profile may be called 6XL, but must specify every head's exact dimensions and calculate parameters, MACs and activation/optimizer estimates; it is not a new production capacity preset or a measured 12-GB fit claim. The teacher is never embedded in the deployed bot.

The future initial student is XXL with unchanged feature schema. Expose configurable alpha and temperature, starting from alpha=0.5 real-label CE plus 0.5*tau^2*KL(teacher||student), with existing head/sample weights preserved. Distill the masked joint-movement distribution and conditional action/target distributions under identical movement contexts; do not reduce teacher output to argmax or independent move marginals. Make teacher calibration and distillation temperature semantics explicit. Offer supervised warmup, distillation and matching QAT stages with reproducible configuration and checkpoint receipts.

Teacher outputs must be generated/read in bounded batches or mmap-backed shards, keyed to checkpoint, dataset/row order, split, features, masks and query contexts. Do not concatenate or duplicate T0011's tens of millions of rows in RAM. Calibration/evaluation must also be bounded. Toy CPU optimization/parity tests are allowed; no real 6XL training, student training, checkpoint selection or large offline dataset run. Treat teacher quality and 50/50 as later validation questions rather than guaranteed improvements.

5. Precision-aware search and caches, disabled by default

Introduce an explicit precision context and selector in the family carriers through their generators. Separate tensor precision from search-node routing. The default selects reference16 everywhere. Experimental routing can choose INT8 for broad successor screening and mixed4_8 for quiet, deeper nodes. Important carrier/death/delivery/drop events, close candidate comparisons and final verification can select the higher-precision profile regardless of depth. Do not implement four-turn expansion here: expose the depth/context seam needed by T0005 and demonstrate it with synthetic states.

Extend cache identity to include checkpoint/model/profile and quantization identity, feature family, perspective, requested-team coverage and parent/history context wherever they are not fixed by the cache lifetime. Preserve full state identity and root immutability. Features and rule computations may be shared; predictions or intermediate activations may only be reused when their complete numerical dependencies match. Switching profiles must not hit another profile's prediction cache.

Provide an optional low-precision shortlist/high-precision verification path for existing candidate evaluation, retaining enough candidates that cheap scores do not irreversibly eliminate every alternative. Compare verified finalists at the same precision with aligned random scenarios and a consistent completed sample count. Discard incomplete rounds; never compare a partly rescored candidate with an unrescored one as though the estimates had identical semantics. Keep a completed incumbent/fallback. Check deadlines before and after atomic prediction work, expose an unmeasured cost-reserve interface and counters, and leave the 2 s startup / 50 ms full-turn limits unchanged. Entropy, score gaps and calibration quantiles are routing heuristics, not certified error bounds.

6. Meaningful correctness checks and later handoff

Add tests for packed round trips and deterministic hashes; scalar/SIMD/profile parity including negative inputs, boundary activations, odd tails and requantization; legal-mask normalization; reference16 compatibility; p2/p3 family/shape rejection; teacher/student conditional alignment and row-identity mismatches; fake-quant/deployment agreement; cache isolation across profiles/history/team coverage; branch order/count root invariance; and deadline cuts preserving completed estimates. Use recorded engine/branch fixtures where already available; do not substitute mock timings for host evidence. Ensure the new optional Torch/QAT contracts execute in an existing Torch-enabled CI job, for example by extending tests.test_torch_backend, rather than only skipping in all core jobs.

Complete core and relevant P-only/derivation/replay tests and document optional-engine/ISA/backend skips. Open a PR linked to this task with the architecture, implemented interfaces, tests and explicit unmeasured limits. Record a future ablation plan: current INT16, uniform8, fixed mixed4_8, adaptive precision plus verification; frozen features/P/V/depth and paired equal-work diagnostics followed by equal-full-turn-time games. Full-model losses/calibration, critical-event errors, candidate ranking/regret, cache hits, search work and full reply faults/timing must all be reported in later evidence.

Dependencies and future work: this code task depends only on the already merged feature/search interfaces (T0009/T0018/T0020), so it can start while T0011 runs. Actual P3 training/export experiments wait for T0011 completion and frozen artifacts. Deployment recommendations require T0013's P3 baseline and later quantization host evidence. Four-step routing integrates only after T0005's depth contracts exist. Any finalist goes through T0006 after all settings are frozen. Propose separate training/host tasks in the handoff; do not claim them, reuse seeds or change existing task definitions implicitly.

## Done when

- A reviewed English architecture in docs/P_MIXED_PRECISION.md, plus runnable experimental export, CPU inference, distillation/QAT and precision-selection code; a prose-only design does not satisfy this task.
- Reference16, uniform8 and mixed4_8 formats/profile manifests have deterministic identities, measured-from-array storage accounting and explicit numeric bounds. Existing INT16 behavior and production model artifacts remain unchanged; all new carrier behavior defaults off.
- Packed INT8/INT4 scalar and supported optimized CPU paths agree with their independent reference/emulation, with honest ISA support/fallbacks. No permanent whole-model INT16 expansion is hidden behind a low-bit size claim.
- Bounded-memory teacher-target and student/QAT interfaces pass toy conditional/mask/row-identity/numeric tests. No real model training or host match/timing batch has been run for this task.
- Precision-aware cache isolation, immutable root/history and completed-candidate/sample/deadline contracts are tested. Existing two-ply seams work; this task does not change search depth or hijack T0005.
- Required core and relevant P-only derivation/replay tests pass with skips documented. The PR links this task and makes no playing-strength or 50 ms certification claim.
- Handoff lists implemented versus deferred pieces, exact regeneration/check commands, numeric/profile identities, remaining CPU/quality risks, and separate future training/host work gated on T0011/T0013/T0005 as appropriate. Shared core and bot code is delivered through a pull request.

## Handoff

*2026-10-11 16:42Z claude-ci@caspers-macbook:*

In review: PR https://github.com/Casper015/ctf-fusion/pull/22 (branch claude-ci/t0030-p-mixed-precision-code-kernels-qat-and, commits 6055584 and 08f07f4 on main 9d5d8e2). Architecture: docs/P_MIXED_PRECISION.md.

Implemented (code only; no real training, export, timing or games):
- fusion/precision.py: profiles reference16 (the existing INT16 export, unchanged), uniform8 and mixed4_8 (squad hidden matrices int4 with 6-bit block scales). Format pprec-1, block 32, proven int32 bounds, requantization onto 0..128, deterministic profile and weight identities, exact emulation, storage measured from the arrays (XXL: uniform8 0.51-0.53 and mixed4_8 0.41-0.43 of the INT16 resident bytes).
- inference/policy_precision/:
  - lowbit.hpp: reference, scalar, SSE2, AVX2 and NEON kernels, block decode, no int16 copy;
  - policy_lp.hpp: p2/p3 heads;
  - bank.hpp: Reference16 is the INT16 path; missing or other-family models fall back, counted;
  - routing.hpp: Selector (off by default), PredictionKey/PredictionCache, RootGuard, shortlist/verify.
- fusion/precision_calib.py: E[x^2] importance on train/val only (test/confirm/holdout/OOD refused), weighted scale search, masked joint/act/target KL per rare context, layer sensitivity, coverage.
- Distillation and QAT: fusion/policy_distill.py, fusion/torch_qat.py, training/distill_policy.py. Loss alpha*CE + (1-alpha)*tau^2*KL (alpha 0.5, tau 1, both unvalidated), mmap teacher shards keyed by rows/masks/contexts, QAT forward equal to the deployed integers, 6XL teacher 5,327,107 params (p2), estimates only.
- training/export_precision.py: headers, npz and manifest under build/p_precision/; bench build and bench-plan; nothing timed.
- Carriers: make_families.py generates ponly_precision.hpp and ponly_p3_precision.hpp; PONLY_PRECISION, PONLY_PRECISION_ROUTE and PONLY_PRECISION_REGISTER; C08's ply-2 cache keys on the profile. With the defaults the code is unchanged.

Tests (2026-10-11, Mac arm64, g++-16):
- core: python -m unittest discover -s tests -t . gave 218 OK (39 skipped: engine/AI_Test/torch).
- torch (CPU venv): tests.test_torch_backend gave 17 OK (5 new QAT tests).
- P-only: discover -s bots/p_only/tests gave 46 OK (3 skipped), plus a rerun of the 4 affected modules after the bundler-define fix: 34 OK.
- Kernel parity: NEON/scalar/reference equal the emulation locally; SSE2 equal under Rosetta (384/384). AVX2 is first exercised in CI (Rosetta has no AVX2).

Commands: see the doc's Commands section. Regenerate the carriers with python bots/p_only/tools/derive/make_families.py.

Not done / limits:
- No quality, CPU, traffic, strength or 50 ms evidence. Sizes come from random weights.
- docs/MARKDOWN_INDEX.md and docs/DIGEST.md do not list the new doc yet: outside these lanes; CI only warns. Run python tools/repo/md_index.py --write in a docs task.

Proposed follow-up tasks (not created; Casper decides):
- A: offline uniform8/mixed4_8 export and calibration report for T0011's frozen XXL P (after T0011).
- B: 6XL teacher, then warmup/distill/QAT of the XXL student with 3 seeds (GPU, after A).
- C: host bench and full-reply timing (after A).
- D: ablation INT16 -> uniform8 -> mixed4_8 -> adaptive+verify, with frozen features/P/V/depth: equal-work diagnostics, then equal-time games (after T0013, B and C; depth routing after T0005).

## Log

- 2026-10-11 08:01Z codex-plan@caspers-macbook: created
- 2026-10-11 08:07Z claude-ci@caspers-macbook: claimed
- 2026-10-11 16:40Z claude-ci@caspers-macbook: amended writes -> CONTRIBUTING.md, bots/p_only/bots/, bots/p_only/include/, bots/p_only/profiles/, bots/p_only/tests/, bots/p_only/tools/derive/, docs/P_MIXED_PRECISION.md, fusion/, inference/mlp.hpp, inference/policy.hpp, inference/policy_p3.hpp, inference/policy_precision/, inference/simd.hpp, profiles/p_precision_code.json, tests/, tools/cpp/, training/
- 2026-10-11 16:41Z claude-ci@caspers-macbook: note: PR: https://github.com/Casper015/ctf-fusion/pull/22
- 2026-10-11 16:42Z claude-ci@caspers-macbook: handoff -> review
- 2026-10-11 17:49Z claude-ci@caspers-macbook: note: CI fix 3b84e39: the Windows job failed on 2 new tests (mapped .npy files kept open, so temp dirs could not be removed; colons from the namespace in the bench binary name). Both fixed; core and torch suites re-pass locally.
