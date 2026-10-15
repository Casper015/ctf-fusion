# Meowmind-inspired large-value on-policy RL pilot

## Decision and status

Casper authorized this new direction on 2026-10-14: build a Meowmind-inspired agent with a substantially larger value network and train it with reinforcement learning. This is a new experiment, not a reopening of retired T0012/T0014 and not admission of an old V model. Production defaults stay unchanged.

At planning baseline `e8b0a22f6365e33d62c2269927af95fc5802c7b6`, the repository has supervised P/V trainers and an offline own-action ranker, but no complete on-policy PPO trainer. The configuration accompanying this plan is a **specification**, not an existing executable training command. Training starts only after the environment/actor/log-probability contracts below pass. No GPU process has been started by this document.

T0051's completed deployment pilot reported negative utility deltas (MLP approximately -0.183, linear approximately -0.156). Those models and their receipts remain controls. Repeated-intervention distribution shift is a plausible explanation, not an established cause. A larger critic alone is not a demonstrated remedy.

## Execution tasks

The [task board](../tasks/README.md) now contains this dependency chain:

| Task | Deliverable | Machine |
|---|---|---|
| T0053 | This plan and declarative pilot configuration | Development |
| T0054 | Native bridge, live rollout adapter and actual PPO trainer | Development; PR |
| T0055 | Correctness smoke and large/small V training pilot | Windows CPU + CUDA; PR |
| T0056 | Actor export, matched control and optional V-guided search | Development; PR |
| T0057 | CPU latency gate and 1176-game primary deployment screen | Windows CPU |

Start T0054 immediately after T0053 lands. T0055 is authorized but dependency-blocked until the trainer passes; it is not a running job. Check and claim the current task before work. Do not create a duplicate task or start old supervised V commands as a substitute. The first host handoff must include the actual tested trainer CLI, not a guessed command. Historical T0051 review can finish independently without changing these experimental results.

## 1. Architecture and attribution

Reference: `bots/meowmind_p/meowmind-ctf-374-20261011.cpp`, original SHA256 `cc7b730cf77c087c8c7f415b3b7d07101c3f003924d3a845a2ea3b81588794bc`. Preserve the original file, attribution and existing artifacts. Its C++ implementation establishes inference architecture, not how its author trained the weights.

| Component | First RL implementation |
|---|---|
| Own actor pi | Shared per-unit/per-move MLP 112 -> 64 -> 32 -> 1, ReLU; warm-start from the reference actor after weight-layout parity tests. |
| Joint movement | Enumerate up to 125 triples. Add the three move logits, mask using the reference's explicit feasibility rules, then normalize over the retained joint actions. Always retain a safe fallback. |
| Attack/pickup resolver | Reference deterministic mathematical assignment, frozen. Its output plus the selected movement is the executed complete order. No new attack/drop exploration in the first trial. |
| Opponent predictor q | Reference Meowmind predictor and online belief update frozen. Adaptive beliefs remain per-episode state and update causally. Do not simultaneously switch to P2/P3. |
| Large V | Public observation features 416 -> 1024 -> 1024 -> 512 -> 1, ReLU hidden layers, linear output: **2,001,921 parameters**, about 69.5 times the original V's 28,801. Initialize fresh; old V values have incompatible units. |
| Small V control | Same 416 features -> 64 -> 32 -> 1, fresh initialization, same training objective/actor/pool/budget. |
| Search | Stage A actor plus resolver only. Stage B adds separately tested value-guided search after Stage A learning and CPU export gates. |

Use a precise architecture ID (`meow_public416_mlp_1024_1024_512_v1`), not an ambiguous "5XL" label. Roughly 8 MB of FP32 weights does not establish CPU inference time. The larger critic is initially a training component; deployment as a search evaluator is a separate gate.

The reference's 416-feature schema is a useful starting point, not complete state information. Explicitly document caps, ordering, padding, HP/carrier HP, flags, time and history. Keep feature definitions unchanged between the large and small V arms. Test feature parity against C++, including enemy carrier HP 100 versus 32. If parity or causal completeness fails, version and fix the schema before freezing both arms.

## 2. What PPO actually learns

Train a policy that controls the same seat throughout complete games. Collect **new on-policy transitions** under the current actor, with full opponent process memory preserved inside each episode. Do not recycle T0010/T0038 branch labels as PPO trajectories.

For the actual sampled joint movement a:

`log pi(a|o) = sum_i logit_i(a_i|o) - logsumexp(valid joint logits)`.

Store the observation, exact joint mask, sampled movement, resolved complete order, old log probability, old value, reward, termination/truncation status, episode identity, actor version and opponent bundle hashes. Define the policy over sampled movements even if two movements resolve to the same outcome; do not deduplicate after sampling and retain the old probability. Unexpected repairs to executed orders invalidate that rollout batch and require a bug fix.

The PPO ratio is `exp(new_logp - old_logp)` for that same sampled action and mask. Search-selected actions must never be scored as though they were sampled from the proposal actor. Training worker policy versions must be explicit: freeze one actor snapshot while collecting each batch; consume that batch, update, then publish a new snapshot. No stale replay in the first pilot.

Stage A therefore optimizes actor-direct play with the resolver. It does **not** claim to optimize the complete searched agent. Stage B initially evaluates search as a separate intervention; if it becomes the behavior policy, collect new data under that behavior and define its action probabilities before continuing PPO. Keep search out of PPO until this contract is implemented.

## 3. Objective, units and critic targets

For final score vector z and learner seat i:

`U_i = sum_{j != i} [1(z_i > z_j) + 0.5 * 1(z_i == z_j)] / (N - 1)`.

Primary reward is zero before termination and `2*U_i - 1` at termination. Use gamma = 1 for the finite game and GAE lambda = 0.95. True terminal states bootstrap zero; rollout-window truncations bootstrap the last value. Runtime failures are recorded as failures, never silently converted to a win, tie or dropped good-looking episode.

First pilot: **no auxiliary kill/pickup/distance reward**, and no hidden privileged-state critic. Optional later potential shaping must be a separately registered arm:

`r'_t = r_t + gamma*Phi(s_{t+1}) - Phi(s_t)`, with terminal Phi = 0.

At gamma = 1 its full-episode sum differs from the original return only by -Phi(s_0). If shaping is introduced, specify public observable Phi and test telescoping. Do not mix the legacy Meowmind hand-potential/capture-score units with a utility-valued critic. Search must use the same reward/discount convention as V. A return critic is not automatically a calibrated probability of finishing first.

## 4. Training settings and bounded execution

Hardware target: Windows host, RTX 4080 Laptop 12 GB; CUDA training, CPU final bot. Start FP32 for numerical parity, then benchmark CUDA TF32/BF16 only as a recorded implementation variant. Keep probability ratios, advantage normalization and loss reductions FP32. Set CUDA flags explicitly. No FP8 assumption.

Initial fixed settings: actor LR 1e-4, critic LR 3e-4, separate optimizers, PPO clip 0.1, value loss coefficient 0.5, entropy coefficient 0.01, gradient norm cap 0.5, 4 optimizer epochs/update, target approximate KL 0.01 with documented early stopping within the update. Normalize advantages per training batch. Value clipping off initially. Start minibatch 1024, rollout target 8192 learner decisions/update. Partial final batches and true terminals must remain valid. Adjust minibatch only for measured memory failures, record the change, and do not silently alter total training budget.

Execution sequence:

1. **Correctness smoke:** synthetic tests plus 20,000 new learner decisions on host. Verify nonzero finite actor/critic gradients, masks, engine outcomes, checkpoint/resume, memory and rollout throughput. This is plumbing evidence, not strength evidence. Save the checkpoint; do not continue a broken run.
2. **Primary large-V pilot:** continue the verified large-V run to **1,000,000 learner decisions total**, counting the smoke decisions. Save fixed checkpoints every 100,000 decisions and a final checkpoint. Predeclare the final checkpoint as the primary screen candidate; intermediate validation is diagnostic, not permission to pick the luckiest test checkpoint.
3. **Small-V ablation:** same total decisions, actor initialization, pool, count distribution and optimizer recipe, fresh game seeds. Use matched curriculum strata; no reusing consumed measurement maps. This isolates critic capacity, not all architectural differences.
4. **Replication:** if implementation is valid, run seeds 102 and 103 for both V sizes as a separately queued fixed-budget replication, regardless of the direction of seed-101's result. A single training seed cannot establish a reliable RL improvement. Optimizer seeds are distinct from game/map seeds.

Initial host authorization covers steps 1-3 (2M learner decisions total across two arms); replication is a follow-up task, not an automatic unbounded sweep. Games with shorter episodes may change episode counts. Budget means learner decisions, never all-seat replies. Report actual wall time, GPU use, CPU throughput and memory; no assumed completion time.

## 5. Opponent pool and data boundaries

Freeze a new training pool from the existing admitted roster, recording exact binary hashes, per-family weights and OOD exclusions. Include production P-only and P+V references where available, but do not promote a known harmful scorer just because it is new. Cap any one family at 15% of opponent seats, verify feasibility, and report realized shares. No learner self-play initially: avoid changing both learner and opponent population at once.

Sample counts 2 through 15 uniformly for the primary objective. Balance learner seats and map symmetries where valid. Record per-count results; a duel-only curriculum cannot support a 2-15-team claim. Freeze an independent validation pool/maps and a final evaluation pool/maps. Held-out opponent lineages stay out of training and tuning.

Reserve fresh seeds in the repository ledger before consuming them. Never reuse T0050/T0051 test maps, the T0037/T0038 corpus, or any consumed measurement seed as a new test. Save per-episode records and rollout receipts; checkpoint optimizer state, RNGs, current pool/curriculum, actor version, consumed episode IDs and unfinished rollout status. Resume exactly, or mark the interrupted batch discarded without reusing its seed.

## 6. Existing components to reuse and missing code

Reuse the authoritative engine, `fusion.runner.Bot` subprocess lifecycle, existing match-log and score verification, task locks, roster identities and evaluation statistics. Inspect their current APIs before implementation. Do not use the independent stress runner with its known first-turn limit bug.

New code lives in `training/meow_rl/`, `tools/cpp/meow_rl/`, `tests/test_meow_rl*.py`, and later `bots/meow_rl/`:

- A causal C++ feature/actor/resolver bridge with explicit reset and per-episode state.
- A live multi-agent environment adapter advancing all orders simultaneously; it must preserve opponent process memory and never leak the current opponents' answers into the actor observation.
- Masked joint actor, separate large/small critic, GAE/PPO trainer, durable rollout store and resumable host entry point.
- Python/C++ feature and actor parity, deterministic replay fixtures and synthetic reward/termination tests.
- An export format distinct from opponent P and historical V artifacts; shape, feature hash and reward-unit checks.

Required tests include legal-mask normalization, impossible-action zero probability, recomputed old logp parity, policy ratio 1 before updates, terminal/truncation GAE, 2-15-team tied-score utility, no future information, reset isolation, native resolver parity, deterministic checkpoint resume and malformed/missing output rejection. The host task must publish an actual tested launch command once the trainer exists. Do not advertise a nonexistent command as runnable.

## 7. CPU search and final evaluation

First export the trained actor-direct bot. Compare it to its identical untrained warm-start actor+resolver control. This isolates RL. Separately compare against unchanged Meowmind and the admitted v12 P-only production control, which answer architectural/product questions rather than isolating RL.

Then add a bounded two-ply search arm using the large V. Cache immutable map geometry and per-observation features, batch leaf evaluations when useful, and key value caches by full encoded state/history and model identity. Keep a legal actor fallback ready before search. No cache may cross different simulated branches merely because positions match. Measure full-turn parse, features, q, search, V and serialization costs on CPU, including first-turn initialization and 15-team tails. Starting with a handful of leaf evaluations is preferable to assuming 2M weights fit the time budget. Distillation is a later independent option if large V is too expensive.

Gate all deployed candidates at 2 s initialization and 50 ms subsequent turns on the host under the registered concurrency. Report p50/p95/p99/max, deadline fallbacks and every failure. GPU training speed is not CPU deployment evidence.

Pre-register the primary screen: final large-V-trained actor-direct versus its warm-start control, equal full-turn limits, paired fresh fixtures, interleaved ordering and an A/A arm. Start a fixed **392 fixtures** (28 independent maps per count, 2-15), three arms including A/A = 1176 games. Use map-cluster paired confidence intervals with equal team-count weight. Secondary Meowmind/production/search/small-V comparisons belong to a separately frozen schedule and require multiplicity correction. Report missing arms and failures, never discard an unfavorable completed fixture. This is a screen, not guaranteed power for small gains.

Only a positive, verified deployment result supports admission. Negative results remain negative; an interval including zero is "not shown". Reduced critic loss, high explained variance or larger networks do not establish stronger decisions.

## 8. Method references

- [PPO, Schulman et al.](https://arxiv.org/abs/1707.06347): clipped surrogate policy updates; this plan's settings are pilot choices, not paper-proven optima for CTF.
- [GAE, Schulman et al.](https://arxiv.org/abs/1506.02438): advantage estimation with a value baseline.
- [Potential-based reward shaping, Ng et al.](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf): basis for the optional telescoping shaping arm, subject to its assumptions.
