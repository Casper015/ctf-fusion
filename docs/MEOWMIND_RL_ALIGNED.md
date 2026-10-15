# Aligned Meowmind RL exports and the search diagnosis (T0061)

Two questions, asked with no new training:

1. **Do the new-pool PPO actors improve CPU deployment?** This compares four actors, all exported the same way.
2. **Why does value-guided search lose?** T0057 measured search − L at −0.148 under the predeclared Bonferroni rule ([MULTIPLICITY.md](../reports/host/meow_rl_screen/secondary/MULTIPLICITY.md)). The candidate causes are quantization, value ranking, and model or simulator mismatch.

T0061 writes the code and the frozen specifications, and tests them without an engine. The host tasks run them, in the order of the [runbook](#host-runbook).

## What is already established, and its limits

- T0057's 4,312 games used **T0055's** checkpoints (pool 1), not T0059's (pool v2).
- Its primary result, L − warm start = +0.0318 [+0.0080, +0.0557], compares two deployed artifacts. The warm start and the candidate were calibrated on different corpora, so it does not isolate PPO's contribution.
- reference16 means INT16 weights, int32 accumulators proved inside ±2^30, and hidden activations clipped to 0..128 (129 levels, about 7 bits). It is not INT8.
- These parts stay frozen for this experiment: the Meowmind-derived actor (112 → 64 → 32 → 1 per unit move, joint over 125 moves at T = 1.05), the opponent predictor and its belief update, the resolver, and the feature schema.
- An identical A/A replay shows the fixtures repeat. It does not show zero sampling uncertainty, or the absence of deterministic bugs.

## 1. One calibration corpus

`bots/meow_rl/aligned/corpus.py` selects decisions from T0059's retained training batches (`runs/meow_rl_pool2/{large_v,small_v}_s101/batches`). It selects 16,000 learner decisions for calibration and 16,000 for export diagnostics, deterministically and before any new game. Each decision carries its 3 × 5 × 112 actor features and its 416 value features. No game is played and no screen or held-out opponent is involved.

**Strata.** Feature indices are zero-based, as written by the reference encoder (`learning::Geometry::policy` and `value`):

| Stratum | Rule |
|---|---|
| Team count | the episode record: 2–15 |
| Phase | turn 1–133, 134–266, 267–400 |
| Carrier | an own unit carries a flag: actor feature 1 of the stay move |
| Combat (proxy) | an enemy is on an attack line of a living own unit: actor features 57/66/75/84 of the stay move |
| Low HP | a living own unit at HP ≤ 34: actor feature 0 ≤ 0.34 (−1 marks a dead unit) |
| Respawn (waiting) | an own unit waits to respawn: value features 11/23/35 > 0 |

Combat is a proxy: it detects an attack line, not an attack. The exact new-life flag cannot be recovered from features, because preparation predicts respawns before encoding.

**Selection rules.**
- **Split.** A map group (team count, map seed) goes to the diagnostic split when `sha256(seed|diagnostic|group) mod 5 == 0`. The group does not depend on which run played the map, so the two splits never share a map, even across the two runs.
- **Quotas.** 1,143 decisions per count for counts 2–13 and 1,142 for counts 14–15, split evenly over the three phases.
- **Order.** Within a cell the selector cycles over the non-empty tactical signatures. Within a signature it takes rows in `sha256(seed|file|episode|turn)` order.
- **Shortfalls.** A shortage is filled from the same count's other phases, then from the other counts. Nothing is duplicated or manufactured, and every shortage is reported.

`MANIFEST.json` records:
- the parent files' sha256;
- every selected (file, row, episode, turn, teams, seat, map seed);
- the requested, available and selected counts;
- the strata;
- the array digests.

## 2. One export configuration for four actors

`bots/meow_rl/aligned.json` freezes the actors, their checkpoint pins (from the committed manifests), the configuration and the experiment-specific variant names. No historical name is reused.

| Actor | Source | Pin |
|---|---|---|
| `warm` | the reference Meowmind actor, no PPO update | reference sha256 `cc7b730c…` |
| `p1_large` | T0055, pool 1, large V | `runs/meow_rl/large_v_s101/checkpoints/final.pt` `f8b98a34…` |
| `p2_large` | T0059, pool v2, large V (**primary**, and the search source) | `runs/meow_rl_pool2/large_v_s101/checkpoints/final.pt` `331ad783…` |
| `p2_small` | T0059, pool v2, small V (capacity ablation) | `runs/meow_rl_pool2/small_v_s101/checkpoints/final.pt` `54acc330…` |

`bots/meow_rl/aligned/export.py` does the following:

1. **Pins.** It refuses a checkpoint whose bytes differ from its pin.
2. **One configuration.** It quantizes every actor to reference16 with the same rules: margin 1.1, ranges rounded up to 8 significant bits, inputs scaled by 2^12 and clamped to ±16384, floor(x + 1/2) rounding. Every actor is calibrated on all 16,000 calibration decisions (240,000 unit-move rows).
3. **Reproducibility.** It quantizes each actor twice and requires the same digest.
4. **Two warm-start routes.** It exports the warm start through two routes: the bridge's reference weights, and the same weights saved and reloaded as arrays. Integer tensors, digest and log-probabilities must be identical.
5. **Agreement on the diagnostic split.** It measures agreement with FP32 on the diagnostic split:
   - greedy joint agreement;
   - total variation and KL of the joint distribution over the feasible mask at T = 1.05;
   - log-probability error;
   - agreement of the sampled joint under four common uniforms per decision (deployment samples, so greedy agreement alone is not enough);
   - saturation per hidden layer.
6. **Native parity.** With a compiler, `tools/cpp/meow_rl/wide_check.cpp` checks the generated header on every diagnostic row, on the scalar and the SIMD path. Its 15 scores, its sampled joint (`sample.hpp`, the deployed draw) and its greedy joint must equal the Python emulation. Each check records the kernel the dispatch actually ran; on the host, `--require-simd avx2` refuses an export whose SIMD checks ran on anything else. The check program switches its standard streams to binary mode on Windows.

All four variants of the search source include **one identical `actor.hpp`**: D (actor-direct), F, R and H. `EXPORT.json` checks this.

Using one corpus controls which data calibrates the export. It does not remove model-dependent quantization error, so that error is reported for every actor.

## 3. Three critic backends (arms F, R, H)

The T0059 large-V critic is `meow_public416_mlp_1024_1024_512_v1`: 416 → 1024 → 1024 → 512 → 1, ReLU, a linear value in 2U − 1 units. `bots/meow_rl/critics.py` builds it three ways from the same calibration value features. Each backend is a header in namespace `meow_rl_critic` with `double value(const float* f)`.

| | F (`fp32`) | R (`reference16`) | H (`wideact16`) |
|---|---|---|---|
| Weights | the checkpoint's float32, unchanged | int16, one power-of-two exponent per layer | int16, one power-of-two exponent per layer |
| Hidden activations | float32 | 0..128 (129 levels, ~7 bits) | 0..32767, int16 storage, 2^15 = 1.0 (32,768 levels, 15 bits) |
| Accumulators | float32, 8 interleaved partial sums, fixed combine order | int32, proved ≤ 2^30 | int64, proved ≤ 2^52 |
| Biases | float32 | int32 | int64 |
| Kernels | `inference/mlp_float.hpp` (scalar, AVX2) | `inference/mlp.hpp` (scalar, SSE2, AVX2, NEON) | `inference/mlp_wide.hpp` (scalar, AVX2, NEON) |
| Parameter bytes (large V) | 8,007,684 | 4,008,964 | 4,019,208 |
| Python emulation | `critics.fp32_forward`, bit-exact | `fusion/quant.py`, bit-exact | `fusion/quant_wide.py`, bit-exact |

### WideAct16, exactly (`fusion/quant_wide.py`, `inference/mlp_wide.hpp`)

For folded layer weights W and biases b:

```
layer 1 (raw integer inputs x, |x| <= 16384):  Wq = round(W / 4096 * 2^a),  Bq = round(b * 2^a)
layer l > 1 (inputs h ~ 2^15 h_real):           Wq = round(W * 2^a),         Bq = round(b * 2^(a+15))
acc   = Bq + sum_i Wq[i] x[i]                   (int64)
act   = 0 if acc <= 0, else min((acc + 2^(s-1)) >> s, 32767)   with s = a - 15 (layer 1) or a (later)
value = acc_out / 2^out_exp                     (an exact int64 -> double conversion)
```

Here round(v) = floor(v + 1/2). The right shift is applied only to positive values, so no implementation-defined shift of a negative number occurs. For acc ≤ 0 every rounding of acc / 2^s is ≤ 0, so the format equals clamp(floor(acc / 2^s + 1/2), 0, 32767) everywhere.

The exponent a is the largest at most 40 that satisfies all of these:
- |Wq| ≤ 32767 (never −32768);
- for every output, Σ|Wq|·x_max + |Bq| ≤ 2^52. x_max is the feature bound 16384 for layer 1 and 32767 after. The sum is computed exactly, and a float estimate before rounding guards against int64 overflow;
- layer 1 needs a ≥ 15, and every hidden layer a shift ≥ 0.

Every partial sum is bounded by the same total, so int64 addition in any order gives the same bits. The SIMD paths multiply int16 pairs into int32 (`pmaddwd`, `vmull_s16`). One pair sum is at most 2·32767² = 2,147,352,578 < 2^31, so it is exact. Each pair sum is widened to int64 before it is added to anything else. There is no SSE2-only path: an x86 CPU without AVX2 uses the scalar loop, and `name()` reports it.

The extra precision is in the activations themselves, not in their storage width. Storing reference16's 0..128 values in an int32 array would not qualify.

### F, exactly (`inference/mlp_float.hpp`)

- **Arithmetic.** Every product and every sum is one rounded float32 operation in a fixed order. A row is summed in 8 interleaved partial sums (lane i mod 8), combined as ((0+4)+(2+6))+((1+5)+(3+7)), then added to the bias. ReLU is y > 0 ? y : +0.
- **No contraction.** Multiply-add contraction is switched off: clang's `fp contract(off)` and GCC's `optimize("fp-contract=off")`. GCC contracts by default on AArch64; without the pragma, native and Python differed by about 8e-7. The AVX2 path uses separate multiply and add intrinsics.
- **Parity.** The native result is therefore bit-identical to `critics.fp32_forward`. Its distance from the training code's float64 forward pass is reported separately; it is about 1e-6 on the plumbing critics.

### The critic scale rule (a finding of this task)

R and H fold the same per-neuron ranges: margin 1.1 × the calibrated FP32 peak, rounded up to 8 significant bits. The rule that built T0057's search critic (`bots/meow_rl/export.py quantize_critic`) gave a never-active neuron the range FLOOR = 1e-3.

That has two consequences:
- Folding by 1e-3 multiplies the neuron's weights by about 1000.
- Every layer has a single weight exponent, and that one row then sets it, so every other neuron loses up to about 10 bits.

The aligned rule gives a never-active neuron the median range of its layer's active neurons. On the plumbing critic this changed the diagnostic-split MAE:

| Backend | MAE, T0056 rule | MAE, aligned rule |
|---|---:|---:|
| reference16 | 0.111 | 0.018 |
| WideAct16 | 0.112 | 0.011 |

On the fixed plumbing trees, reference16 under the T0056 rule (R0) picked F's action on 76.7% of roots; R and H matched 100%.

These are **plumbing numbers on a synthetic critic, not evidence**. The host diagnostics measure the real critic. T0057's deployed critic had a mean error of 0.21 under the T0056 rule. `reference16_t0056` (R0) is therefore kept as a diagnostic-only backend, so that R relates to the critic T0057 screened. R0 is never deployed.

An INT64 accumulator alone does not repair clipping. Saturation counts are reported per layer for R, H and R0.

## 4. Search v2 (`bots/meow_rl/search_v2.hpp`, `MEOW_RL_SEARCH=2`)

v2 keeps v1's two-ply rule unchanged:
- 4 root candidates: the actor's sampled choice, then the top other joints by accumulator;
- 2 opponent scenarios;
- two simultaneous plies on the reference's World;
- our ply-2 continuation is one draw of the integer actor;
- value = terminal 2U − 1 or V(leaf), with gamma 1;
- the choice is the highest scenario mean, ties to the actor's order.

`search.hpp` (v1, T0056/T0057) is unchanged and still built by `MEOW_RL_SEARCH=1`. Two things change in v2.

- **Common random numbers.** Every stream is a function of (game seed, turn, scenario) only:
  - the ply-1 opponent orders, drawn once per scenario and shared by every root;
  - both simulator seeds;
  - the ply-2 opponent orders: `Planner::rng`, set before each use;
  - the continuation uniform.

  A turn's tree therefore depends only on the observed state and the INIT, not on earlier searches or on the candidate order. Two facts make this hold:
  - `Planner::rng` is read only by `enemyOrders`;
  - `shots()` reads only the probabilities it is given, and `predict()` without saving overwrites only per-turn scratch that each later call recomputes.
- **Staged admission.** The clock starts after the block is read. Work is admitted only if its estimated cost fits before `MEOW_RL_SEARCH_DEADLINE_MS` (30 ms; 1,500 ms on turn 1):
  1. before the setup (root ordering and the scenarios' opponent orders), the setup plus one trajectory;
  2. before every trajectory, the whole trajectory;
  3. before the ply-2 expansion (successor forecasts, 15 continuation actor evaluations, opponent orders, ply 2), the expansion;
  4. before the critic call, one critic call;
  5. after the evaluation, the deadline itself.

  A trajectory refused part-way still records what it spent. The reply is resolved after the search, and its cost is measured too (`max_resolve_us`).

  An estimate is max(prior, the largest cost seen this game) × 150%. The priors per backend are in `aligned.json`. Any refusal discards the whole search and plays the actor's choice, which is already computed; a partial comparison is never used. This is an empirical gate, not a real-time proof: the eight-worker preflight shows whether any reply was late. The MEOWRL counter line adds:
  - trajectories;
  - refusals per stage (`refuse_trajectory`, `refuse_expand`, `refuse_critic`, `refuse_late`);
  - the largest setup, trajectory, expansion, critic, search, resolve and turn costs (µs);
  - the first-turn cost;
  - the critic format.

`tools/cpp/meow_rl/tree.cpp` builds the same trees through the same functions with no deadline. A test shows the deployed v2 bot's choice equals the harness tree scored in Python, for every backend. On the plumbing check this held on 58 of 58 roots for each of F, R and H.

## 5. Fixed-work diagnostics (`bots/meow_rl/aligned/trees.py`, `diagnose.py`)

1. **collect** (host, exactly 8 workers, a fresh seed base in `docs/MEOWMIND_RL_SEEDS.md`). Diagnostic games of the deployed D bot against the training lineages of pool v2, 16 per count, one count per game. They are never strength evidence and never use the held-out roster. Each seat's stream is rebuilt exactly from the match record: its INIT plus the shared TURN blocks, which the runner checked against what the seat received.
2. **select** (before any tree is scored). 1,024 roots: 74 per count for counts 2 and 3, 73 for 4–15. Within a count they split evenly over the phases; turn 1, with its 1,500 ms deadline, is excluded. Within a phase they cycle over the tactical signatures, at most 6 roots per game. The roots come from fresh diagnostic maps, never from the calibration corpus's maps.
3. **build**. One materialized tree per root. Each tree records and hashes (FNV-1a 64):
   - the feasible-joint mask, the actor's uniform and choice, and the candidate keys;
   - the turn seed, and every scenario's streams: both opponent RNG states, both simulator seeds and the continuation uniform;
   - the hash of each scenario's ply-1 opponent orders;
   - for every leaf: terminal flag, plies, reward, the resolver's own actions at the root and at the continuation, the continuation joint, the hash of the ply-2 opponent orders, and the 416 value features.

   Reduced trees (fewer than 4 feasible candidates) are counted.
4. **diagnose**. Every leaf is scored by F, R, H and R0. With a compiler, the native headers must reproduce every leaf bit for bit. The report covers, against F:
   - leaf MAE, RMSE, max and bias, and saturation;
   - top-action agreement;
   - discordant candidate pairs;
   - F's regret of each backend's choice;
   - agreement by F's top-two margin and by count;
   - how often each backend overrides the actor.

   95% intervals come from a bootstrap over games, because roots of one game are not independent. `ROOTS_SCORED.jsonl` keeps every root's candidate values and choice per backend, so any disagreement can be inspected.

This measures the critics' arithmetic on identical trees. It cannot tell inaccurate value ranking apart from inaccurate opponent or simulator forecasts. That needs real-engine continuations: a separate task with its own seed ledger and budget, not authorized here.

## 6. The eight-worker gate (`bots/meow_rl/aligned/preflight.py`)

The gate runs **exactly eight concurrent game workers**: not eight threads per bot, and not six workers.
- **Scope.** All seven distinct variants (deduplicated by binary sha256) play the same 64 fixtures: 16 maps at counts 2, 5, 8 and 15 against the held-out roster, at 2,000 ms first and 50 ms later replies. That makes 448 games, interleaved fixture by fixture.
- **Pass rule.** A variant passes only if all of these hold:
  - no seat faults, replies late, sends a malformed reply or dies, and our bot refused no block;
  - counters are present for every turn played;
  - the steady maximum is below 48 ms and the first reply below 1,800 ms;
  - a search variant completed at least one search.
- **Recorded, not gated.** Admission costs and refusals are recorded, and compared with the priors.
- **Receipt.** It records:
  - the host: power scheme, GPU activity before and after, thread environment;
  - the compiler and flags;
  - every bundle and binary sha256, and the search settings;
  - every game, with first and steady timing kept separate.
- **On failure.** If any arm fails, the gate is **blocked**. The failure is kept. Concurrency is never lowered and an arm is never dropped to make it pass. A new preflight needs a fresh seed and new frozen settings.

`run_screen.py` now enforces four things:
- it refuses a config whose seed is null;
- it refuses a `--workers` override when the config freezes `workers_exact`;
- it refuses an arm whose binary is not the one the receipt measured;
- with `freeze`, it refuses unless the config file and its schedule are byte-identical to what the freeze recorded, and the schedule carries the config's seed, counts, map budget and roster.

## 7. Schedules A and B

Both use the held-out roster: Fable, MiMo, Kimi, GLM, Player4. `bots/meow_rl/aligned/freeze.py` audits it against both training pools by name, path and source sha256, including every file each bot includes locally. It scans all of them for the reference's distinctive identifiers and records the executable the runner actually starts. A hit, or a missing executable, blocks the freeze until a human decides. The scan is a screen, not a proof of independence.

The freeze then:
- writes each fresh seed into its profile;
- generates each schedule once;
- records every frozen identity in `FREEZE.json`: the profiles' and schedules' sha256, the arms' binaries, families, workers and budgets.

| | Schedule A ([profile](../profiles/meow_rl_aligned_screen_a.json)) | Schedule B ([profile](../profiles/meow_rl_aligned_screen_b.json)) |
|---|---|---|
| Arms | warm, p1_large, p2_large, p2_small, + A/A of p2_large | D, F, R, H, + A/A of R |
| Fixtures, games | 392 (28 maps × counts 2–15), 1,960 | 392 on a separate seed, 1,960 |
| Workers, limits | exactly 8; 2 s / 50 ms | exactly 8; 2 s / 50 ms |
| Family | primary p2_large − warm at 95%; secondary p2_large − p1_large and p2_large − p2_small, Bonferroni over 2 (z = 2.2414) | F−D, R−D, H−D, H−R, F−R, Bonferroni over 5 (z = 2.5758) |

The two schedules total 3,920 formal games. On top of that come the preflight (448) and the diagnostic collection (224 games), both recorded separately.

Rules for both schedules:
- No early stopping or extension.
- Failed fixtures and missing arms are kept and reported.
- `bots/meow_rl/aligned/report.py` judges every comparison under its family, with computed quantiles. Results are reported in percentage points of pairwise rank utility. A comparison with any scheduled fixture missing gets the verdict "incomplete", never a confirmatory one.
- An interval containing 0 is not shown, never "equivalent".
- B's arms do whatever work fits in the frozen deadline, so B compares deployment packages. The fixed-work diagnostics compare precision.
- A one-seed large-versus-small or old-versus-new-pool difference is evidence about those checkpoints, not a general effect.

## Host runbook

These steps run on desktop-g60ojud with `CTF_CAN=host,cuda,ai-test`, after T0061 is merged. Wrap every timed or collecting step in `python tools/repo/task.py lock cpu --task <T> -- …`, with CUDA training and benchmarks stopped. Reserve every seed in its ledger before its first game, and check `docs/DIGEST.md`.

```bash
python bots/meow_rl/aligned/corpus.py --batches runs/meow_rl_pool2/large_v_s101/batches runs/meow_rl_pool2/small_v_s101/batches --out data/meow_rl_aligned/corpus
python bots/meow_rl/aligned/export.py --corpus data/meow_rl_aligned/corpus --require-simd avx2
python bots/p_only/tools/build.py --stats --variants bots/meow_rl/variants_aligned.json
python bots/meow_rl/aligned/trees.py collect --seed-base <S_roots> --out data/meow_rl_aligned/trees
python bots/meow_rl/aligned/trees.py select --out data/meow_rl_aligned/trees
python bots/meow_rl/aligned/trees.py build --out data/meow_rl_aligned/trees
python bots/meow_rl/aligned/diagnose.py --trees data/meow_rl_aligned/trees --corpus data/meow_rl_aligned/corpus --out reports/host/meow_rl_aligned/diagnostics
python bots/meow_rl/aligned/preflight.py --seed <S_preflight>
python bots/meow_rl/aligned/freeze.py --seed-a <S_A> --seed-b <S_B>
python bots/p_only/tools/run_screen.py --config profiles/meow_rl_aligned_screen_a.json
python bots/p_only/tools/run_screen.py --config profiles/meow_rl_aligned_screen_b.json
python bots/meow_rl/aligned/report.py
```

**What to commit.**
- the corpus manifest, as `reports/host/meow_rl_aligned/CORPUS.json`;
- `inference/meow_rl/aligned/` (headers, manifests, `EXPORT.json`) and `bots/meow_rl/variants/aligned/`;
- the diagnostics, `PREFLIGHT.json`, `FREEZE.json`, the screen reports and `VERDICTS.*`.

**What stays out of Git.** Corpus arrays, trees, match logs and checkpoints stay in `data/` and `runs/`, with backups whose copied hashes are verified. A second disk on the same host is not an off-machine backup.

**What not to conclude.**
- "RL gain confirmed" from the historical mixed-calibration screen.
- "Zero noise" from A/A.
- "Large V mechanism proved" from one training seed.
- "Strongest bot" from the highest sample mean without the comparison that tests it.
