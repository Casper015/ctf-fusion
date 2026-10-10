# Feature families, richer V and the 4XL V pilot

This note covers task T0009, requested on 2026-10-10 against main at 7f30c80, and the work it hands to the GPU and host tasks.

It describes four things:
- why the opponent policy P and the value network V get new inputs, and which inputs;
- how the inputs are versioned, so that old weights can never meet new features;
- how the V data and the controlled 4XL pilot are set up;
- how a large V would be deployed under the 50 ms deadline.

Generated details live elsewhere: every feature, range and scale is in [training/specs/FEATURES.md](../training/specs/FEATURES.md), and the experiment parameters are in [profiles/v_pilot.json](../profiles/v_pilot.json) and [profiles/p_features.json](../profiles/p_features.json).

**Status:** implemented and tested on a development machine. Nothing here is trained, and nothing is demonstrated in games. [Status](#status) at the end lists what was checked where.

## 1. Diagnosis

### P and V read different inputs

| Net | Input | Width |
|---|---|---:|
| P mover, per unit and move | global, unit k, move (k, m) | 40 |
| P act, per unit and move | the same as mover | 40 |
| P squad, per team | global, three units, 15 moves | 222 |
| P target, per unit, candidate and move | dist, hp, carrying, support, the hit bit of the move, attacked_by_team, target_home | 7 |
| V, per team | global, three value units, 12 aggregates | 54 |

`encodePolicy` encodes the predicted team from its own perspective. That team's units' hit points, carried flags and positions are therefore already P unit features: P does see the predicted carrier's own HP. A test now pins this down (`test_predicted_team_hp_reaches_legacy_unit_features`).

### Where each stored field is consumed

The extractor record stores more than any one head reads. A field in the record is not necessarily an input of a given head.

| Record block | Consumed by |
|---|---|
| `g`, `u` | mover, act, squad |
| `m` | mover and act (the chosen move's row), squad (all 15) |
| `cand_f` | the target head only, and only columns dist, hp, carrying, support, the hit bit of the chosen move, attacked_by_team and target_home |
| `legal`, `act_legal`, `cand` | masks only (joint, act, target) |
| `origin` | labels only (requested move blocked by a collision) |
| `v` | V |

### The verified gap is between heads

A target's hit points, carried flag and delivery distance reach only the target head. Nothing tells the mover, squad or act head whether a destination puts a one-hit kill, a carrier or a coordinated volley in reach. The decision to approach, retreat, attack or pick up cannot depend on it.

`test_target_hp_reaches_the_combat_block_only` checks this directly. An enemy at range 2 with 100 hp and with 34 hp gives identical p2 mover and squad inputs. Only the target head's candidate features differ, and in p3 the combat block does too: `kill1_stay` goes from 0 to 1 at the staying and the approaching move.

### The candidate limit

The target head sees at most the 8 nearest hittable enemies within Manhattan 4. `fusion/labels.py` keeps an attack on a hittable target outside the slots as an attack without a slot, so that target label is lost.

The limit is unchanged. [`training.feature_audit`](../training/feature_audit.py) measures, per team count:
- the share of attack labels lost this way;
- how often all 8 slots are full.

Run it on the layout-3 shards before deciding anything about the limit.

### Concrete V2 aliases

These are pairs of states with identical v2 inputs and different futures:

| Alias | Why v2 cannot tell them apart | v3 feature that separates it | Test |
|---|---|---|---|
| An enemy carrier at 100 hp and at 34 hp | V2 has no enemy hit points at all | `ec0_hits` (3 against 1) | `test_v2_alias_separated_by_v3` |
| A dropped flag that returns in 14 turns and one that returns in 1 | Only counts of dropped flags | `dropped_return_min` | `test_v2_alias_flag_return_timing` |

v2 also cannot see:
- cooldown timing;
- which rival holds a carrier, and how its score compares with ours;
- whether our units can reach a firing cell on a carrier (`carrier_shots` is now-only);
- whether our own units are exposed to fire from shooters that move first (`threatened_by` is now-only).

`feature_audit` also measures how much outcome variance sits inside groups of identical v2 inputs across matches, and how much of it v3 separates.

## 2. Feature families and versioning

A model reads one family. The record layout is versioned apart from the families.

| Family | What it reads | Implementation |
|---|---|---|
| p2 | the legacy policy inputs | `inference/features.hpp` `encodePolicy`, `inference/policy.hpp` |
| p3 | p2 plus the combat block per (unit, destination): mover/act 55, squad 447; the target head unchanged | `inference/combat.hpp`, `inference/policy_p3.hpp` |
| v2 | the 54 legacy value features | `encodeValue`, `inference/value.hpp` |
| v3 | v2 plus the 54-feature extension (108) | `inference/combat.hpp`, `inference/value_v3.hpp` |
| layout 3 | the layout-2 blocks at unchanged offsets, then `pc` (3 × 5 × 15) and `vx` (54); 923 int16 per record | `tools/cpp/extract_features.cpp` |

### Family identity

A family's identity is the digest of exactly what it reads:
- the group specs: names, bounds and scales, in order;
- the block shapes;
- the head-input composition;
- for p3 and v3, `kCombatVersion`.

Offsets and other families' features are not part of it. A new V family, or a wider layout, leaves every P artifact valid. `fusion/families.py` holds the definitions. The legacy digests p2 `213481118d8a6263` and v2 `ff0f693ef8e95ff0` are frozen there, and a test checks that the extractor still reproduces them.

### Checks

Every check fails closed. None was weakened to make old artifacts pass.

| Where | Check |
|---|---|
| A training run (`train_policy`, `train_value`) | Records `features: {family, digest, layout_version}`. Warm starts refuse another family or digest. |
| Shards (`training/dataset.py`) | Accepted for a family when their stored schema provides it with the same digest at the same offsets, whatever their layout version. Layout-2 shards still serve p2 and v2, and are refused for p3 and v3. Appending to a dataset still requires the identical schema. |
| Export (`training/export_weights.py`) | Recomputes each run's family digest. The artifact identity of a new family carries the family digests. A legacy pair keeps its old identity: `spec_version` plus the layout-2 schema digest `d3813c50359fb9cf`, so gate reports made before families still verify. |
| Header | A p3 or v3 header includes `policy_p3.hpp` / `value_v3.hpp`, uses `PolicyModelP3` / `ValueModelV3` and static_asserts every first-layer width against `kMoverInP3`, `kSquadInP3` or `kV3`. P2 features cannot reach a p3 model: no conversion exists, a test compiles the attempt and expects a failure. A p2 model given p3 features reads exactly its p2 part. |
| Registry, mixed export | One registry holds one family pair; `ModelSet` points at that pair's types. Mixed exports check both sides' digests, so a frozen P can be paired with a V of a newer family without retraining P. |
| Carriers | `static_assert` that the pre-included registry's types are the families the carrier was compiled for. |
| Records (`fusion.native.extract`) | Alignment is checked from the stream's first turn; a branch successor starts at turn t. |

### Legacy outputs are unchanged

- The layout-2 columns of the new extractor are bit-identical to the extractor at 7f30c80 on the recorded 2-, 4- and 15-team games; their digests are frozen in the tests.
- No existing header changed: features.hpp, policy.hpp, value.hpp, adapter.hpp and mlp.hpp are untouched, and so is every bot bundle.
- Legacy exports, mixed sets and registries are written exactly as before; `tests/test_mixed_export.py` re-derives the committed mixed sets.
- `kSpecVersion` stays 2. It now names the legacy encoders that p3 and v3 extend.

## 3. The p3 combat block

The block holds 15 features per own unit and destination, all from the engine's rules:
- damage and hit points come from INIT;
- the hits that kill a unit are ceil(hp / damage);
- hits add up within a settlement;
- a unit respawning in this settlement can neither shoot nor be shot;
- a carrier cannot shoot, because attacks settle before drops;
- line of fire is `Board::canAttack` from the post-move cells.

Two kinds of quantity are named apart:
- **`*_stay`** features are facts about the current positions. They hold if the named enemies stay, since their orders are unknown.
- **`*_reach`** features are upper bounds over every legal move of the units named.

Movement blocking is not modelled. No future action or outcome is an input.

| Features | Meaning |
|---|---|
| `kill1_stay`, `kill1_carrier_stay` | Enemies one hit kills from this destination, and how many of them carry a flag. |
| `volley_stay`, `volley_carrier_stay` | Enemies this unit plus the teammates able to reach a firing cell can kill this turn. |
| `focus_present`, `focus_hits`, `focus_home`, `focus_support` | The carrier in range with the fewest hits (then the nearest delivery). Its hits, delivery distance and support stay attached to the same unit. |
| `chase_present`, `chase_dist`, `chase_hits`, `chase_home` | The nearest enemy carrier within 6 (pursuit), with its own attributes. |
| `exposure_reach`, `lethal_stay`, `lethal_reach` | Enemy shooters able to fire at the destination after their move, and whether the staying or reaching shooters can kill this unit. |

**Missing values.** An illegal move, or a unit off the board, has an all-zero block, as its legacy move features do. Own-fire features are 0 when the unit cannot shoot. Every attribute of an absent focus or chase carrier is 0, behind its `*_present` bit.

**Cost.** `CombatSummary` is computed once per state and shared by every team's encoding and by the v3 extension. A per-cell threat map makes `exposure_reach` constant time per destination. Buffers are sized once per board; `runtime_check` proves that nothing allocates after turn 1.

The P-only views compute one summary per built state, root or branch. That cost is part of what the equal-time screen measures. `tools/host/value_cost.py` times p2 and p3 extraction and the joint head with real weights.

**Kept as it was:** the target head, the candidate limit, the adapter, search depth and the mixture.

## 4. The v3 extension

The 54 features, grouped (full table in FEATURES.md):

| Group | Features |
|---|---|
| Own units (3 × 3) | Hits that kill it, enemy shooters able to reach a firing cell, whether that kills it. |
| Enemy carriers (2 slots × 8, nearest delivery first) | Present, hits, delivery distance, our shooters now and by reach, killable by a volley, escort, its team's score gap. |
| Own carrier | Escort of the carrier nearest delivery. |
| Flags (7) | Soonest return of a dropped flag, soonest cooldown end, our and their nearest flag distance and the difference, dropped flags each side reaches first. |
| Rivals (3 slots × 5, by score) | Present, score gap, carriers, active units, nearest carrier's delivery distance. |
| Races (6) | Own and enemy units respawning, rivals ahead or tied/one behind that hold a flag, enemy carriers we can kill, own carriers that can die this turn. |

The total is 108 inputs, inside the provisional 96–128 range. Every slot exists because one of the aliases above needs it. No input pads the width.

## 5. V data under a frozen policy

**The target** is the expected final pairwise utility of our seat under a stated continuation policy and opponent distribution. It is not the probability of finishing first.

**The freeze**, before any V data:

| Item | Frozen as |
|---|---|
| Carrier | `v12_ponly_xxl` |
| P | the committed XXL, family p2, every team-count model |
| Search | fixed-time v12 settings |
| V during collection | off |
| Opponents | the reference roster frozen in the collection schedule |
| Data seed | 20261201: training only, never a screen seed |

v12 is the pilot carrier because it is what the request names and because the stage 2 screen gives no reason to prefer C08: its V paths are thinner (no duel path) and its search fills the time budget. **Every V capacity comparison uses this same frozen P and carrier.**

**Collection.** [bots/p_only/profiles/collect_v_data.json](../bots/p_only/profiles/collect_v_data.json) is a one-arm screen with `keep_matches`: 500 maps for each of the 14 team counts. Then:
- `training.dataset --rows designated` keeps only the carrier's own rows;
- `train_policy` refuses such shards;
- our own carrier, family "candidate", never becomes an opponent-policy label.

**Refresh.** Once a V changes the carrier's play substantially, that carrier is a new policy. Version it, and collect again.

**Old trajectories.** Old mixed-roster trajectories can pretrain V explicitly (`--init`). Their outcomes are not outcomes under the frozen carrier, so the pilot's primary arms do not use them.

**Splits.** By map, as everywhere: a map's matches, roots and branches share one split. Reserved opponent families stay out of the data roster, and the confirmation seed stays unchosen.

### Branch continuations

These are the ranking data ([training/branch_collect.py](../training/branch_collect.py)). For a root, turn t of a frozen-carrier match:

1. Rebuild the engine from the match seeds and step it with the recorded orders. Every reproduced block must equal the recorded one, or the root is dropped.
2. Start every bot fresh and feed it exactly the recorded inputs, so its history is the root's. Its replies are discarded; their differences from the record are counted as `prefix_mismatch`.
3. At turn t, the opponents keep their recorded replies (they could not see our order), and our seat plays a candidate.
4. From turn t + 1, everything is live under the official limits, to the end.

**Candidates** are the carrier's own reply plus distinct one-unit move deviations: legal orders, not predictions. P-generated guesses are never continuation labels.

**Root integrity.** The root match is only read. A test checks that its bytes and the reproduced root state are unchanged after branches.

**Caveat.** The carrier's internal memory at turn t reflects its own choice, not the forced candidate.

**Ranking data.** [`training.branch_dataset`](../training/branch_dataset.py) encodes each candidate's successor with the bot's encoder and pairs candidates within a root:
- a pair is resolved when the outcome difference exceeds one pairwise step (and two standard errors with repeats);
- near ties stay uncertain, with no label and no weight;
- each root counts once.

## 6. The 4XL pilot

The arms differ only in feature family and hidden widths:

| Arm | Family | Hidden | Parameters (3 training heads) |
|---|---|---|---:|
| A | v2 (54) | 256 / 32 | 22,403 |
| B | v3 (108) | 256 / 32 | 36,227 |
| C | v2 (54) | 8192 / 512 | 4,646,915 |
| D | v3 (108) | 8192 / 512 | 5,089,283 |

The committed M value (v2, mixed-roster data) is an extra reference.

**Held equal across arms:** the same designated rows, targets, splits, evaluation and procedure. The pilot uses one seed per arm; finalists use three independent seeds. The model is one shared `general` network conditioned on the team count, not one per count.

**Training** (`training.train_value`):
- torch on the RTX 4080 Laptop, FP32 reference (TF32 and mixed precision off);
- unchanged semantics: clipped ReLU, weighted BCE over heads with the auxiliary weight, Adam with coupled L2 on matrices only, seeded minibatch order;
- validation-based stopping with plateau decay;
- `--checkpoint` writes the best state with its epoch and validation loss.

**Data volume.** `training.value_cache` keeps only the family's value columns, memory-mapped; the torch backend streams bounded windows.

**Batch size.** `training.value_sweep` measures rows per second and peak allocated/reserved VRAM at batch 2048, 4096 and 8192 for the 4XL shapes, and records out-of-memory instead of retrying. Parameter storage is small (about 20 MB in FP32); activations and data are what the sweep measures. Nothing here promises a training time or a speedup.

### Decision quality

[`training.evaluate_value`](../training/evaluate_value.py) scores every run on the same rows:
- outcome log loss, Brier score, expected calibration error and the calibration table;
- the same by phase and by team count;
- subsets: carrier, drop, interception, close score;
- the learning curve and training effort;
- with branch data: pair accuracy and estimated decision regret, next to the on-policy and the uniform choice's regret.

`--compare` gives paired differences with 95% intervals clustered by map. Correlated turns never count as independent samples.

**Ranking supervision** is optional (`--rank`, `--rank-weight`): the pairwise logistic loss on the outcome logits. The outcome loss stays the primary objective and the stopping criterion.

**Auxiliary heads.** The capture-within-20 and death-within-10 heads remain as controls. No delivery or interception head was added: "any capture in 20 turns" is not an individual carrier's delivery probability, and such a head needs its own episode and censoring definitions first.

## 7. Deployment on the CPU

**The tail dominates.** In a dense 4XL pass, the 8192 × 512 tail is 4.19 M multiply-adds:
- 90% of the pass with 54 inputs;
- 83% with 108 inputs.

First-layer caching cannot remove the tail.

**The earlier measurement does not apply.** The historical 191 ms result enlarged P and V together under a 16-process load. It is not a V-only measurement.

### Route A: direct, bounded

The family carriers (`bots/p_only/tools/derive/make_families.py`, which leaves every existing carrier byte-identical) evaluate the exported V on the settled successors they already score. Two settings bound it, and both are off by default:
- `PONLY_V_MAX_CALLS` caps network V calls per turn;
- `PONLY_V_DEADLINE_MS` stops them after that many milliseconds of the turn. The 4XL arms set 30 ms, a tuning parameter.

A decision path whose V was cut takes the same path's choice without V, so no decision mixes candidates scored with and without V. With zero calls allowed, the replies equal the V-off build on a recorded game; a test checks this.

**Size.** A 4XL V is about 35 MB of header text. It is written once per model set and shared by all 11 team-count models (`export_mixed --value-run`). The bots need `FUSION_MAX_WIDTH` 8192, which the registry sets.

### Route B: distillation

[`training.distill_value`](../training/distill_value.py) trains a student on three signals:
- real outcomes, the primary target; validation, stopping and calibration use only them;
- the teacher's calibrated probability, with a configurable weight;
- optionally, ranking pairs.

The student must read the teacher's family: it cannot reproduce what its inputs omit. Any shape is allowed, for example M-sized, or a wide first layer with a small tail, whose cost a first-layer accumulator mostly removes.

### What to measure on the host

| Measurement | Source |
|---|---|
| Per-call view, encoding, full and incremental pass; first-layer share; inputs changing between calls; weight bytes; p2 and p3 policy extraction with real weights | `tools/host/value_cost.py`, at 1, 6 and 15 concurrent copies |
| Full-reply p50/p99/max, deadline faults, V calls and seconds, cut calls and fallback decisions | the V pilot screen, under `PONLY_STATS` |
| Search samples lost to V | the stats of C08 V arms, if any are added |
| Startup and first-turn time | the first-reply maximum |

The 50 ms reply limit, the 30 ms search budget, the 35 ms compute target and the 40 ms output cutoff are unchanged.

## 8. Evaluation plan

| Step | Where | What |
|---|---|---|
| P offline | GPU task | p2 and p3 at XXL capacity, same shards, configuration and seed (`profiles/p_features.json`). `training.compare_policy`: joint, act and target log loss by contact, pursuit, retreat, own carrier, carrier shot, distance band and team count, with map-clustered intervals. |
| P in games | host task | `screen_p_features.json`, V off, equal full-turn time, tuning seed 20261210, A/A copy. |
| V offline | GPU task | The four arms, then a ranking arm and the distilled student; `training.evaluate_value`. |
| V in games | host task | `screen_v_pilot.json` at fixed P: V off, the existing V, the new small V, direct 4XL and the distilled V, with A/A. Tuning seed 20261211; coefficients are tuned on it only. |
| Confirmation | T0006 | Finalists frozen first. At least 10,000 independent map groups over counts 2–15, at least 500 per count, three independently trained seeds, paired fixtures, the official rating definition and anchor. |

Rules for every step:
- An interval that includes zero is reported as "not shown".
- Lower prediction loss and larger models are not, alone, evidence of stronger decisions.

## Status

### Implemented

- Families p2, p3, v2 and v3, with compatibility checks and the legacy migration.
- Layout-3 extraction and FEATURES.md.
- Family-aware training, evaluation, exports, mixed exports and registries.
- Designated-seat V data.
- Branch continuations and ranking data.
- V-only caches, ranking loss, checkpoints, sweeps, evaluation, distillation.
- The P comparison and the feature audit.
- Family carriers with the bounded V route; the V cost bench.
- Variants and profiles.

### Tested on the development machine (not timing or strength evidence)

- The repository suite and the P-only suite, plus the AI_Test-backed pipeline tests.
- The torch-backend parity tests (p3 batches, the v3 ranking loss, sweep bookkeeping, teacher inference) on CPU PyTorch 2.14.1. The CUDA path is checked on the host with `FUSION_TEST_DEVICE=cuda`.
- Legacy columns bit-identical on recorded games.
- p3/v3 C++ and Python parity through quantized models, no allocation on the runtime path, and an exact v3 accumulator.
- The combat block recomputed from the engine's rules on recorded games.
- V2 aliases separated by v3.
- Masks, immunity and respawn.
- Incompatible combinations refused, in Python and at compile time.
- Family carriers keep every P-only contract (root history, branch parents, coherent states, engine comparison) with p3/v3 sets.
- The V-cut equivalence.
- The 4XL quantization bound (a test). A 4XL v3 header (35 MB) was compiled once by hand; no test compiles one.
- Branch roots reproduced exactly and never changed.

### Not done here

- Nothing is trained.
- No feature or model is shown to help in games.
- No timing was measured.

These are the dependent GPU and host tasks.
