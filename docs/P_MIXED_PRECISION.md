# Mixed-precision opponent policy P: kernels, calibration, distillation, routing

This is the infrastructure for running the opponent policy P with lower-precision weights. It is code and contracts only (task T0030).

Not done in this task:
- no model was trained, distilled or quantized from a real checkpoint;
- nothing was played or timed on the host;
- every bot default is unchanged.

The production path is the existing INT16 export (`fusion/quant.py`, `inference/mlp.hpp`). It stays bit for bit what it was. The family carriers select it unless they are built with `PONLY_PRECISION=1` and `PONLY_PRECISION_ROUTE=1`.

P predicts what the **opponents** do, inside the C08 or v12 heuristic search. It does not choose our own moves. Features, search depth, candidate generation, V, evaluation coefficients and time budgets are all frozen here.

## Where things are

| File | Responsibility |
|---|---|
| [fusion/precision.py](../fusion/precision.py) | The three profiles: per-layer weight bits and identities. Low-bit quantization: row scales, int4 block scales, bias, requantization, proven bounds. Packing, flattening and digests. Exact integer emulation. Storage measured from the arrays. |
| [fusion/precision_calib.py](../fusion/precision_calib.py) | Activation second moments (importance). Weighted reconstruction error. Masked joint, act and target KL against reference16, per rare context. Per-layer sensitivity. Coverage of rare contexts. |
| [fusion/policy_distill.py](../fusion/policy_distill.py) | NumPy distillation: teacher distributions; the alpha·CE + (1−alpha)·τ²·KL loss with exact gradients; memory-mapped teacher shards and their identity checks; 6XL teacher estimates. |
| [fusion/torch_qat.py](../fusion/torch_qat.py) | Torch QAT. The forward pass computes the deployed integers exactly; gradients flow through a float surrogate (straight-through). Also the QAT and distillation loss. |
| [training/export_precision.py](../training/export_precision.py) | Deterministic low-bit headers, `arrays.npz` and `MANIFEST.json` under `build/p_precision/`. The benchmark build and plan. |
| [training/distill_policy.py](../training/distill_policy.py) | Bounded-memory entry point: teacher targets, then the warmup, distill and QAT stages with receipts; the `toy` contract run. |
| [inference/policy_precision/lowbit.hpp](../inference/policy_precision/lowbit.hpp) | Low-bit layers. An independent per-weight reference, a scalar path, and SSE2, AVX2 and NEON kernels with runtime dispatch. Requantization. |
| [inference/policy_precision/policy_lp.hpp](../inference/policy_precision/policy_lp.hpp) | Joint, act and target heads on low-bit nets, for p2 and p3 alike (the existing input builders). |
| [inference/policy_precision/routing.hpp](../inference/policy_precision/routing.hpp) | Profile, NodeContext and Selector. PredictionKey and PredictionCache. RootGuard. Shortlist and verification. |
| [inference/policy_precision/bank.hpp](../inference/policy_precision/bank.hpp) | One policy in several precisions, and the dispatch. Reference16 calls the INT16 functions. |
| [bots/p_only/tools/derive/make_families.py](../bots/p_only/tools/derive/make_families.py), [search_experiment.py](../bots/p_only/tools/derive/search_experiment.py) | Generate `ponly_precision.hpp` and `ponly_p3_precision.hpp` (a precision-aware View). Add `PONLY_PRECISION` to the family carriers, and the profile to C08's ply-2 view cache. |
| [profiles/p_precision_code.json](../profiles/p_precision_code.json) | The declared profiles, calibration rules, distillation start values, routing defaults and future ablation. |
| [tools/cpp/precision_check.cpp](../tools/cpp/precision_check.cpp), [precision_routing_check.cpp](../tools/cpp/precision_routing_check.cpp), [precision_bench.cpp](../tools/cpp/precision_bench.cpp) | Kernel and policy parity driver; routing contracts on synthetic states; benchmark driver (built, not run). |

Data flow:
1. A float PolicyNet goes through `precision.quantize_policy(profile, importance)`.
2. The result has integer arrays per head and layer.
3. `export_precision.header` turns them into constexpr arrays plus one `fusion::PolicyModelLP`.
4. A variant registers that model in the carrier's `PolicyBank` (`PONLY_PRECISION_REGISTER`).
5. `View::profile`, chosen by the Selector, dispatches each head.

`precision.forward_acc` is the oracle for every C++ path, and `torch_qat` reproduces it.

## Formats and their numerics

The three profiles share shapes and features:

| Profile | Weights | Use |
|---|---|---|
| `reference16` | the existing INT16 export: power-of-two shifts, int16 weights | the default; also the high-precision verifier |
| `uniform8` | int8 everywhere, one float64 scale per output row | screening |
| `mixed4_8` | int8, except the squad head's hidden matrices: packed int4 with 6-bit block scales | quiet, deeper nodes |

A low-bit layer (bits 8 or 4; 16 is allowed for a sensitive layer kept wide) is integer from input to accumulator:

```text
w_eff[o][i] = d[o][i/32] * q[o][i]     q: int8 [-127, 127], int4 [-8, 7], int16 [-32767, 32767]
                                       d: 1, except int4: 1..63 per 32-column block
acc[o]      = B[o] + sum_i w_eff[o][i] * x[i]                 int32
hidden      h = clamp((int64(acc) * M[o] + 2^(R-1)) >> R, 0, 128)
last        real = double(acc) * out_scale[o]                 out_scale = S[o] * s_in
```

Here `s_in` is 1 for raw features and 1/128 for hidden activations, `M/2^R ≈ 128·S·s_in`, R ≤ 46 and M < 2^31.

**Activations.** Hidden activations stay on the existing clipped-ReLU scale 0..128, which fits uint8 but not int8. First-layer inputs are the extractor's signed integers and travel as int16, as in the INT16 path.

**Bounds.** The exporter proves per row that `bound = Σ|w_eff|·x_max + |B| ≤ 2^30`, with x_max from the feature spec (first layer) or 128 (hidden). If a large bias would break the bound, it widens the row scale.

**Why every path gives the same bits.**
- Every partial sum (one block, d times a block, any summation order, a pair of `pmaddwd` products) is bounded by the row bound, so int32 never overflows and integer addition is exact in any order.
- The requantization product stays below 2^61 in int64. C++20 makes `>>` an arithmetic shift, matching NumPy's.

**Rounding.** Every rounding is floor(x + 0.5), in NumPy, Torch and C++.

**The output scale is rounded on its own.** A compiler allowed to contract (`-ffp-contract`) would otherwise fuse it into the next addition as an FMA and change the last bit; a volatile store in `lowbit::toReal` prevents this. The INT16 path never had the issue, because its output scale is a power of two.

**Packing.** A block holds 32 columns; inputs are zero-padded to whole blocks.
- int8 and int16 rows are padded to `in_pad`.
- In an int4 block, byte j holds `q[j]+8` in its low nibble and `q[j+16]+8` in its high nibble. That is the llama.cpp Q4 layout: one AND, one shift and one subtract decode 32 weights.
- Padding weights are 0. Block scales are stored as uint8.

**Decoding happens in registers.** The kernels decode packed weights block by block: sign extension (`pmovsx` or SSE2 unpack, `vmovl`) or nibble unpacking, then the existing 16-bit multiply-add. No int16 copy of a low-bit matrix is ever made.

**Identity.**
- `profile_identity` is a hash of the format, the profile and the block size.
- `weights_digest` hashes every exported array: names, dtypes, shapes, values.
- The header embeds both (`kManifestIdentity`, and the PolicyModelLP `identity` and `profile`).
- The manifest also records the header's SHA-256.
- The native build cache keys on every source it includes (`fusion/native.py`).
- Experimental exports live only under the ignored `build/`.

### Storage at the XXL shape

Measured from the exported arrays of a random-weight XXL policy (mover 256, squad 1024/256, act 256, target 32). "Packed weights" includes the zero padding, which the padding column counts separately.

| Family | Profile | Resident bytes (weights, biases, scales) | vs INT16 | Squad head (bits per layer) |
|---|---|---:|---:|---|
| p2 | reference16 | 1,094,796 | 1.00 | 1,048,564 (16/16/16) |
| p2 | uniform8 | 574,808 | 0.53 | 535,268 (8/8/8) |
| p2 | mixed4_8 | 451,928 | 0.41 | 412,388 (8/4/8) |
| p3 | reference16 | 1,570,956 | 1.00 | 1,509,364 |
| p3 | uniform8 | 804,184 | 0.51 | 764,644 |
| p3 | mixed4_8 | 681,304 | 0.43 | 641,764 |

**Other costs.**
- Padding: small first layers waste the most (mover and act, 40 inputs padded to 64: 6,144 bytes each in p2).
- Scales and biases: about 3% of the low-bit totals.
- Scratch per forward: two int16 activation buffers and one int32 sum buffer of the widest padded layer (8,192 bytes for the squad head).
- Resident copies: one per array. Nothing is expanded.

**A smaller file is not a speed claim.** CPU cost depends on decode work, cache residency and the 16-bit multiply-add, and has not been measured.

## Calibration and mixed precision (IQ and AWQ ideas)

Inspected revisions of llama.cpp (ggml-org/llama.cpp):

| File | Commit | What it showed |
|---|---|---|
| `ggml/src/ggml-quants.c` | 2923cf2862ad0afa159444cf07fec7600d755fe1 | `make_qx_quants` and `make_qkx*` search a scale over ±9 perturbations, minimizing importance-weighted error. `quantize_row_q4_K_impl` weights each weight by `imatrix · sqrt(σ² + x²)`, and its 6-bit sub-block scales are quantized against a row super-scale. |
| `tools/imatrix/imatrix.cpp` | 26908739bc8ab9c5ccacbde5dc9c0b3609079e8f | Per input column it accumulates Σx² of the activations, with counts. |
| `src/llama-quant.cpp` | 649dcb1036fd47d01a83b56c5e35913e65c277ba | Per-tensor type rules keep sensitive tensors at more bits (`use_more_bits`, attn_v, ffn_down). |

AWQ ([arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) supplies the principle: protect the weights of large-activation inputs.

What is built here:
- **Importance.** `ActivationStats` streams E[x²] per layer input over bounded batches: raw integers for first layers (the folded weights multiply those), 0..128 for hidden layers. It runs on declared train or validation rows and optional recorded or synthetic successor states, and counts each source.
- **Refused splits.** Test, confirmation, holdout and OOD splits are refused, so scales, routing thresholds and temperatures never see them.
- **Weighted scale search.**
  - int8: the row scale is chosen from 21 multiples (0.70–1.10) of the max-abs scale, minimizing Σ imp·(w − w_eff)².
  - int4: per block, the integer d around the max-abs choice. This is the deployed integer d, so the error searched is the error shipped.
- **Diagnostics.**
  - Importance-weighted reconstruction error per layer.
  - KL(reference16 ‖ profile) of the masked joint, of act at the recorded moves and of target under the same move contexts. Reported overall and for six rare contexts: carrier, low HP (≤ 34), pickup, drop, delivery (carrier within 2 of home) and combat.
  - Per-layer sensitivity: one layer fake-quantized at a time, the others float.
- **Protection.** Protection is by whole layers or 32-column blocks (`mixed4_8` keeps every first and last layer, and the small act and target heads, at 8 bits). There is no sparse per-weight exception path. Which layers deserve protection is for the sensitivity report on real checkpoints to decide. No layer is declared critical in advance.

**Where an IQ3-style codec would fit.** It would replace the int4 block format: 3-bit indices into a lattice codebook (IQ3_XXS/S) with per-block scales. For these small MLPs the costs differ from an LLM's:
- the blocks would be 256 columns, wider than most layers here;
- the padding to whole blocks is proportionally large;
- decoding goes through a table, not shifts.

It would need its own quality measurement (the distributions above) and its own CPU measurement. A 3-bit file is not a 3-bit speed.

## CPU paths and ISA support

| Path | Where checked | Status |
|---|---|---|
| reference (per-weight decode loop) | everywhere | the oracle of `tools/cpp/precision_check.cpp` |
| scalar | everywhere | equal to the emulation in tests |
| NEON (AArch64) | macOS arm64 (development machine, CI macOS job) | equal to the emulation |
| SSE2 (x86-64) | locally under Rosetta 2; CI Linux jobs | equal to the emulation (384 of 384 outputs locally) |
| AVX2 (x86-64, runtime check) | CI Linux runners when their CPU has AVX2 | Rosetta does not expose AVX2; first checked in CI |

`FUSION_SIMD=0` forces the scalar path, as for the INT16 kernels. No timing has been measured on any ISA.

## Distillation and QAT

Nothing was trained. These are interfaces with toy CPU tests.

**Loss, per head, with the existing head weights (act 0.5, target 0.5) and sample weights:**

```text
alpha * CE(student, recorded label) + (1 - alpha) * tau^2 * KL(teacher_tau || softmax(student / tau))
```

alpha = 0.5 and tau = 1 are configurable starting values to validate later.

- **Joint head.** It distills the masked 125-way joint, never argmax and never independent marginals.
- **Act and target heads.** Both are distilled at each unit's recorded move, with the same legal masks for teacher and student.
- **Teacher temperature.** Teacher probabilities come from its calibrated logits (its fitted beta) divided by tau.
- **Student temperature.** The student's raw logits are divided by tau inside the KL. Its own temperature is fitted after training, on validation rows.

**Bounded memory.**
- Row caches are directories of memory-mapped `.npy` arrays, read one minibatch at a time in a seeded permutation.
- T0011's `data/pf_policy_cache/<split>` has no `row.npy`; its `CACHE.json` receipt (rows, shapes, dtypes, data SHA-256 per array) stands in. The row ids are the positions in that fixed order, and the cache identity is derived from the receipt's digests. `python -m training.distill_policy verify-cache --cache DIR` rehashes every array against the receipt (task T0031).
- A teacher shard is refused on a cache other than the one it was written for (dataset identity and row count).
- Teacher shards are preallocated memmaps written batch by batch. They hold the masks and movement contexts and an `IDENTITY.json`: teacher state digest, family, split, tau, teacher beta, dataset, and the row, mask and context digests.
- A student batch whose row ids, move contexts or any mask differ from the shard's is refused. An incomplete shard cannot be closed.

**Stages** (`training/distill_policy.py`):
1. warmup: supervised.
2. distill.
3. qat: the same loss through `fusion/torch_qat.py`.

warmup and distill run in NumPy by default. With `--backend torch --device cuda` they run the float student on the device (`fusion.torch_qat.FloatPolicy`, features built on the device from the int16 records), with the same loss. qat always runs on Torch. Torch and NumPy losses and gradients agree within float tolerance (`tests.test_torch_backend`).

Each stage writes `config.json`, `receipt.json` and `model.npz`.

**QAT forward.** It quantizes the current weights with the deployment's max-abs rule (Torch, operation for operation), then runs the integer pipeline exactly: integer products summed in float64, which is exact because every sum is ≤ 2^30 < 2^53, and requantization in int64. Gradients flow through the straight-through estimator, with clipped-ReLU masks.

**Two consequences.**
- "Fake quantization" is the deployed arithmetic, so the QAT forward equals the NumPy emulation bit for bit (tested).
- The exported beta is fitted after QAT and folded into the last layer.

**Teacher profile 6XL** (teacher only, never exported). Estimates from `python -m training.distill_policy teacher-profile`, not a measured fit:

| Head | Dimensions (p2) |
|---|---|
| mover | 40 → 1024 → 1 |
| squad | 222 → 4096 → 1024 → 125 |
| act | 40 → 1024 → 4 |
| target | 7 → 128 → 1 |

| Profile | Parameters (p2) | fp32 Adam state (params, m, v) | Activations kept, batch 4096 |
|---|---:|---:|---:|
| 6XL teacher | 5,327,107 | about 60 MiB | about 837 MiB |
| XXL student | 545,443 | about 6 MiB | about 213 MiB |

The student is XXL with the unchanged feature schema.

## Precision routing, caches and verification (disabled by default)

**Tensor precision is separate from search routing.**
- A `Profile` says which weights compute a prediction.
- A `NodeContext` says why it is asked: depth below the real root, carrier, death, delivery or drop events, close comparisons, final verification, screening.
- The `Selector` maps context to profile:
  - disabled (the default): Reference16 everywhere;
  - enabled: root, important events, close comparisons and final verification get `high` (reference16);
  - quiet nodes at depth ≥ 2 get `mixed4_8`;
  - screening successors get `uniform8`.

**Depth is only a routing input.** The four-turn expansion is T0005's: `NodeContext.depth` is the seam it will fill, and the routing check exercises it on synthetic states.

**Caches.** `PredictionKey` holds everything a prediction depends on that a cache lifetime does not fix:
- model (weights) identity and quantization (profile) identity;
- profile, feature family and perspective (our team);
- requested-team coverage;
- state and parent (history) hashes.

A hit also requires full equality of state and parent, so a hash collision cannot alias two states. Features and rule computations may be shared; predictions only when every numerical dependency matches.

In the generated carriers:
- a View computes all its heads at one profile, fixed per build (`View::profile`, and `View::used` after any fallback);
- C08's ply-2 view cache reuses a view only at the profile it was built with.

Switching profile never hits another profile's prediction. `RootGuard` records the real root and its history; branches predict from copies with the root as parent.

**Shortlist and verification** (`routing.hpp`):
- The finalists are the completed incumbent, the `keep` best cheap scores, and anything within `closeGap`. Cheap scores never eliminate every alternative.
- Finalists are verified at one precision, round by round on aligned scenarios: round r uses the same scenario for every finalist.
- The deadline is checked before each atomic prediction (with the cost reserve, an interface not a measured value) and after it.
- An incomplete round is discarded whole, so means always cover the same completed rounds.
- With no completed round, the incumbent is returned and counted.
- Counters: screened, finalists, rounds started/completed/discarded, predictions, deadline cuts, fallbacks.

The 2 s first-turn and 50 ms full-turn limits are unchanged. Entropy, score gaps and calibration quantiles a carrier might use for routing are heuristics, not error bounds.

**In the carriers.**
- `PONLY_PRECISION=1` makes `make_families.py` include the generated precision View. Its `profile` defaults to Reference16, and the bank's Reference16 entry is the carrier's own INT16 model.
- `PONLY_PRECISION_ROUTE=1` enables C08's selector for ply-2 successors (depth 1, screening).
- `PONLY_PRECISION_REGISTER(b)` lets a variant register low-bit models. A missing or other-family model falls back to INT16 (counted).

With the defaults, the carriers' code is unchanged by the preprocessor.

## Commands

```bash
python -m unittest tests.test_precision                      # formats, kernels, policies, calibration, distillation
python -m unittest tests.test_torch_backend                  # QAT contracts (needs torch; the CI torch job)
python -m unittest bots/p_only/tests/test_precision_carriers.py   # carriers: disabled = before; routed contracts
python bots/p_only/tools/derive/make_families.py             # regenerate ponly_precision*.hpp and the family carriers
python -m training.export_precision export --policy RUN --profile mixed4_8 --name NAME [--importance imp.npz]
python -m training.export_precision bench-plan --header build/p_precision/NAME/mixed4_8/policy_lp.hpp
python -m training.distill_policy teacher-profile
python -m training.distill_policy toy [--torch]              # every stage on synthetic rows
```

The benchmark command `bench` builds `tools/cpp/precision_bench.cpp` against one header. On the host, a later task runs it with `--run` and recorded feature rows. It reports:
- the cold first call and the warm per-call time of each head;
- weight and activation bytes per call;
- together with `bench-plan`'s CPU, compiler, flags, concurrency, header and profile identities.

## What is not measured, and what comes next

**Not measured, and not claimed:**
- the quality of any profile on a trained checkpoint;
- CPU time or memory traffic on the host;
- playing strength;
- whether 50/50 distillation helps;
- whether 6XL fits in a given GPU.

Sizes are from arrays of random weights.

**Proposed separate tasks.** None is claimed by T0030.

| Task | Waits for | Work |
|---|---|---|
| A. P precision offline | T0011's frozen checkpoints | Calibration stats on train/val, sensitivity, and export of uniform8 and mixed4_8 for the frozen XXL P (p2, and p3 if it is the finalist). Report the distributions per rare context and the storage. |
| B. Teacher and distillation | A, the GPU | Train 6XL on the bounded caches. Write teacher shards. Run warmup → distill → QAT for the XXL student, with three seeds. Report validation loss and calibration against the INT16 student. |
| C. Host cost | A | Run `bench` and the full-reply timing under the host's concurrency. Measure kernels, startup and warm replies, and weight and activation traffic. |
| D. Ablation | T0013's p3 baseline, B, C; T0005 for depth routing | Frozen features, P, V and depth throughout. Arms: current INT16 → uniform8 → fixed mixed4_8 → adaptive precision with verification. First paired equal-work diagnostics, then equal-full-turn-time games. |

Task D must report:
- full-model losses and calibration;
- critical-event errors;
- candidate ranking and regret;
- cache hits;
- search work;
- full-reply faults and timing.

Any finalist then goes through T0006.
