---
id: T0065
title: "Redesign CPU bot architecture around existing P (not refactoring)"
status: review
owner: claude-review@caspers-macbook
needs: []
writes: [bots/agent_v2/, inference/agent_v2/, training/agent_v2/, tools/cpp/agent_v2/, tests/test_agent_v2.py, tests/test_agent_v2_training.py, profiles/agent_v2.json, docs/AGENT_V2.md]
depends: [T0061]
land: pr
branch: claude-review/t0065-rewrite-modular-cpu-carrier-around
updated: 2026-10-16
---

## Goal

### Mandatory scope: architecture redesign, not refactoring

**THIS TASK IS AN ARCHITECTURE REDESIGN, NOT A REFACTOR.** Casper explicitly requires a new architecture designed around the current goals. Do not treat the task as extracting, renaming, wrapping or reorganizing the old Meowmind architecture while preserving its structure or behavior. Reuse a module only when it earns its place in the new design; existing code and checkpoint compatibility are optional assets, not constraints. Design the new module responsibilities and data flow first. Retain the mathematical tactical approach and our P where useful, while replacing the old predictor/habit pipeline and unsuitable coupling. Judge success by the new specification, official rule correctness, training/deployment consistency, CPU limits and subsequent measured strength, not resemblance to the old bot. Historical source and evidence remain comparison references.

### User-approved direction clarification (2026-10-15 Pacific)

Casper explicitly clarified: this is an architecture redesign, not a behavior-preserving refactor. The new carrier is NOT required to reproduce the original bot's full replies, action choices, tactical scores, feature schemas or search trajectory. These requirements supersede the earlier migration-parity wording. Preserve game semantics, causal inputs, compatible reused-model contracts, numerical correctness and training/deployment consistency. Keep the mathematical tactical layer and MLP direction; replace the old opponent expert/habit system with our P. Preserve legacy artifacts as external controls. The incumbent task owner/branch/status and current code work remain assigned; reconcile implementation with this clarification without resetting work.

Build a simple, modular CPU carrier that reuses ctf-fusion's learned opponent P, existing PPO actors/critics, mathematical tactical assignment and measured inference kernels. This is a user-requested architecture rewrite for future learning experiments, not a CNN project. First deliver a tested redesigned carrier and an explicit new behavior/training contract; host training and strength runs follow separate frozen tasks.

### Intent and current evidence

Casper requested this task on 2026-10-15 (America/Los_Angeles). Retain the mathematical layer and the MLP direction. Replace the original Meowmind opponent predictor and its hand-written online expert/habit updater in the NEW carrier with our existing P. Preserve the original source, existing bots, checkpoint artifacts and completed evidence as controls.

Historical planning snapshot was main 58de8cf. Read the latest T0062-T0064 status; completed old-carrier results are historical controls and diagnostics, not a behavioral specification for the redesign. Preserve their frozen inputs and consumed seeds. New-carrier precision, training and search choices require validation on the new carrier.

The previous T0057 search had negative utility; a rewritten search is experimental until evaluated. A mathematical DP is exact for its configured local score, not for winning the game. Existing learned P files are reusable artifacts, not proof that every P size or family is a good choice for this carrier.

### Target architecture

```text
Public input + causal history
              |
       State / map cache
         /           \
 own actor pi       our opponent P
         |            |         \
 joint moves     move marginals  joint moves + conditional actions/targets
         \          /                       |
       tactical resolver                    |
              |                             |
     ready complete default order           |
              |                             |
      optional bounded search <--- sampled opponent orders
              |
     candidate-local simulation -> V evaluation
              |
 complete chosen order / ready fallback -> protocol output
```

Keep the public runtime interface small: initialize/reset one episode; decide from one observed turn and a deadline; return complete orders and telemetry. Internal modules vary only at actual seams: predictor, actor, resolver, simulator and value evaluator. Use compile-time selection or lightweight concrete adapters where practical; no generic plugin system, virtual dispatch framework or new service dependency.

Two modes share the same state, P, actor and resolver: actor-direct (V used during learning, absent from its decision path) and bounded search (V evaluates simulated successors). Both must use the same opponent P. Do not silently retain the old Meowmind predictor in the resolver or deeper search plies.

### Reuse / extract / remove map

| Existing asset | Decision |
|---|---|
| `inference/policy.hpp`, `policy_p3.hpp`, their feature encoders and pinned P2/P3 model registries | Reuse through an explicit adapter. P2 XXL is a concrete reusable reference, not a mandatory final choice; existing P3 and distilled P3 remain named alternatives, not assumed winners. No new P training here. |
| P's joint 125-move distribution, move marginals, action and target heads | Use marginals for the resolver and joint/conditional distributions for enemy scenarios. Retain squad coordination instead of independently sampling three movement marginals. |
| `training/meow_rl/` PPO, GAE, actors/critics, storage, pool/seed handling | Reuse tested logic. Add a new environment/bridge adapter and run identity without silently changing an old run's code or resume contract. |
| Warm-start, T0055 and T0059 actor/V checkpoints | Preserve and load only with exact compatible features, units and hashes. They are initialization/reference artifacts after a behavior change, not automatically admitted policies. |
| Meowmind geometry, prepared-state conventions, 112 actor features and 416 V features | Reuse useful geometry and feature code with attribution. Actor/V feature schemas may change when justified by the new design; version them and reject incompatible weights. Reused frozen P inputs must still satisfy their original encoder contracts. A shared public state does not mean forcing P and pi/V into one feature vector. |
| `Planner::shots` 8-subset DP | Extract into a resolver taking explicit state, moves and opponent move marginals. Retain the mathematical assignment approach. Original coefficients and tie order are an available reference configuration, not an acceptance constraint on the new design. Document/freeze any scoring changes and never label them engine rules. |
| Official rules and existing simulators | Retain authoritative semantics. Reuse or repair one simulation module after parity checks; do not write a new learned world model. |
| `inference/mlp.hpp`, `mlp_float.hpp`, `mlp_wide.hpp`; aligned export/calibration | Reuse numerical implementations and integrity checks. Choose production precision from evidence, not format names. |
| Search v2 common random numbers, staged budgets, fixed-tree harness and telemetry | Reuse as optional planning machinery, parameterized on the new predictor/resolver/state interfaces. It must never call the original predictor behind those interfaces. |
| Existing screen, all-seat fault accounting, exact-eight-worker preflight and frozen schedules | Reuse tooling; register new variants and fresh schedules for this carrier. Do not mutate T0062-T0064 inputs. |
| Original Meowmind learned enemy experts, heuristic expert mixture, evidence arrays and hand-tuned habit updates | Remove from the new production carrier once P replacement is complete. Keep the old implementation in historical reference builds; a legacy-predictor adapter is optional and must not delay or constrain the new runtime. |
| Original Meowmind own-policy inference inside opponent prediction | Remove duplicate evaluations from the new path. The current actor provides our behavior; any own-team marginal needed by an enemy action resolver must come from a documented current-policy forecast. |
| Original hand-potential V/reward, old deep-beam control flow, monolithic renamed-main include | Exclude from the new production carrier. Preserve historical implementations for tests/attribution. Use reward units consistent with the trained V. |
| Failed offline own-action scorer, new CNN/Transformer, new giant teacher, full autoregressive attack policy | Exclude from this task. They do not solve the migration contract. |

### Prediction interface contract

Audit `inference/policy.hpp` and `policy_p3.hpp`: P is an OPPONENT model, not our movement actor. Existing P2/P3 has a squad residual plus per-move action/target conditionals. Its target cap is currently eight slots per unit: map slots to stable world unit IDs and record truncation/missing support rather than inventing targets.

A prediction result identifies state, history, observed turn, feature schema, team, model/calibration digest and precision. It exposes normalized joint movement probabilities; consistent per-unit marginals; conditional action probabilities; and conditional target distributions with explicit slot IDs. Document the factorization between enemy teams and conditional heads. Do not claim that separately trained action heads encode arbitrary cross-unit attack correlations.

Preserve the existing encoder's raw-versus-prepared state timing, movement numbering, team perspective, dead/respawning units, HP, flags, legal-mask meaning and history. Do not erase collision mass merely because our actor avoids collisions: opponents can request colliding actions, and the engine resolves them. Distinguish invalid protocol actions from modeled orders with no effect.

History updates once per REAL observation. A branch gets an isolated hypothetical history view; never update persistent adaptation/history using simulated events. Removing the old habit updater does not mean removing raw causal history required by P or the actor. Any future learned temporal adaptation is a separately trained model version.

Reuse per-team P outputs within exactly the same state/history. Root predictions cannot be blindly reused after a branch changes HP, death, flag possession or timers. Lazy conditional heads, preallocated buffers and immutable map caches are welcome; any approximation or team-skipping rule must be explicit and separately measured. No silent old-predictor fallback when the new P is expensive.

### Resolver and action contract

The resolver consumes a joint move and opponent movement marginals, then returns complete own-team actions. Keep exact engine range/line-of-sight, action eligibility and settlement semantics. The subset assignment and its scoring are tactical choices: reuse or improve them under a documented new configuration. Original constants may remain as an optional reference profile, but legacy score/tie-order equivalence is not required.

Be precise about current limits: the resolver normally returns attack targets or pickup/wait; it does not currently generate voluntary drops. Merely exposing the X token is not a learned drop policy. Adding a legal drop candidate is optional, not a prerequisite; specify any change explicitly and validate it rather than claiming the existing DP already supports drops. Do not conflate existing self-collision avoidance with an engine rule; the actor's feasible mask is a policy restriction and must be documented as such.

Complete candidate orders, not just movement keys, are passed to the simulator. Preserve exact flag semantics: attacks happen before pickup, pickup before voluntary drop, and drop before capture. A carrier cannot drop and fire in one turn, and a newly voluntarily dropped flag cannot be picked up in that same pickup phase. Cover these with engine-based fixtures before adding any later drop experiment.

### Flag randomness and simulation contract

The official environment has N+1 designated refresh spots and max(2, ceil(N/2)) flags. Free-spot selection excludes spots occupied by home flags; dropped flags remain on their cells. Return and cooldown timers are 15 and 8 turns. Verify all details against the engine pinned by the run rather than assuming versions never change.

Map seed, flag seed and actor sampling seed have distinct roles. The live actor never receives hidden flag RNG state or future flag spawns from logs. Search samples lawful refresh outcomes and handles multiple refreshes in engine order. Its RNG is a scenario generator, not the actual hidden engine RNG. Common random seeds across roots reduce variance but do not guarantee identical refresh events when candidate actions change RNG consumption or free spots.

Reuse map geometry and deterministic rule calculations. Check simulated transitions against the real engine, isolating stochastic refresh cases from deterministic parity. Cover all team counts 2-15, terminal timing, respawn immunity, movement collisions, attacks, pickup/drop/capture and return/cooldown order.

### Implementation sequence and attribution of changes

1. Inventory exact source/model paths and licenses/attribution; write `docs/AGENT_V2.md` with an architecture diagram and concrete interface contracts. Freeze rule/contract fixtures and model identities; old-bot complete-order agreement is not an acceptance gate.
2. Implement the new public state/history, geometry/features, actor and mathematical resolver around the target design. Validate against official engine rules and the new interface specification. Local equivalence tests are required only for implementations explicitly reused unchanged. Do not build a legacy adapter solely to reproduce old complete decisions.
3. Integrate a pinned existing P through the new predictor interface. Reused P must match its own encoder/inference implementation on identical inputs, independently of old bot decisions. Wire joint movement and action/target conditionals into the new resolver/scenario generation. The first new carrier may intentionally differ in several design choices; document them and establish a new baseline rather than forcing a one-change migration.
4. Connect optional V/search through the same state and prediction interfaces. Use fixed work before equal-time gates. Keep branch state and randomness explicit. Preserve a complete actor-direct order before expensive search; record partial-work refusal and fallback.
5. Add an environment/bridge adapter for future PPO on the new carrier. It must execute the same P/resolver as the native deployment path. Reuse existing GAE/PPO/checkpoint code and build identity checks. Validate finite synthetic updates and log-probability recomputation without launching a full training run.
6. Deliver the single-file bundler/export configuration and a concrete host handoff with new variant names, commands, pinned models and required evaluation contrasts. Create dependent host validation/adaptation tasks through task.py when the code and relevant T0063/T0064 evidence are available; do not silently start matches or training as part of this code task.

No production source may include the entire original Meowmind .cpp with main renamed after extraction is complete. Preserve only the required, attributed implementations plus separately retained legacy fixtures/adapters. Reuse existing inference/trainer modules rather than copying whole parallel stacks.

### Subsequent learning, specified here but not bundled into the rewrite

- First establish a rule-correct, reproducible new-carrier baseline. Compare its complete bot against historical bots as external strength controls, without requiring matching actions. Isolate individual new design changes in later ablations where useful. Both actor-direct and search need measured CPU feasibility; do not assume our P is cheaper.
- Once an opponent P is selected from evidence, freeze it and collect fresh PPO trajectories on the new runtime. Old actor/V weights are optional warm starts only when input/output/normalization/reward contracts match; otherwise initialize new networks and train fresh. Do not reduce the new design to fit old checkpoints. Do not resume an old run while changing the environment silently.
- A small zero-initialized joint movement residual is the first optional actor extension: sum of existing per-unit scores plus delta(state, joint move). This is a separate ablation with its own identity, not required to preserve any legacy behavior. Reuse the concept of the P squad head, not opponent P weights as an own actor without training.
- Resolver coefficient selection and a drop/keep head remain conditional follow-ups after real-engine tactical diagnostics show headroom. A few discrete profiles are sufficient for a first experiment; Gaussian coefficients and full autoregressive actions are not required.
- CNNs and new teacher sweeps remain out of scope.

### Resource and evidence requirements

Final deployment is pure CPU, first turn 2 seconds and all later full replies 50 ms. Allocate immutable geometry/model buffers during initialization when useful; thread-local/per-branch scratch must remain isolated. Memory availability does not make stale cache reuse correct or guarantee 2-second loading. Include every served count/model in startup cost.

Every future official host match/preflight uses exactly EIGHT game workers under the CPU lease, with GPU training stopped. Freeze variants and settings before results; failure at eight stays failure, never silently reduce to six. Future CUDA training uses its own task and leases. No host timing claims from development machines.

Keep old evidence, raw logs and checkpoint backups. Shared calibration, sampling policy, model-family identities, data lineage, all-seat failures, search counters, actual work, paired map-cluster intervals and multiplicity stay explicit. Local numerical equivalence checks support safe reuse, not strength. Changed complete decisions are expected and establish neither improvement nor harm without matched new-carrier evaluation.

## Done when

- A source-backed reuse/extract/remove inventory and architecture diagram in docs/AGENT_V2.md, including P versus own actor roles, legacy reference scope and the next learning tasks.
- A new single-file CPU carrier built from modular attributed code; actor-direct and optional search share one predictor/resolver path. No production dependency on the original expert mixture/habit updater or renamed-main monolith.
- Correct P/model interfaces and numerical equivalence for code explicitly reused unchanged; versioned new actor/V inputs and actions; mathematical resolver tests against its NEW declared scoring/assignment rules; complete-action protocol tests. NO legacy full-action, reply, old scoring or tie-order parity gate.
- Explicit causal-history, branch-isolation, target-slot, probabilities/marginals, cache identity and state-timing tests; simulator parity with the pinned official engine and lawful hidden flag randomness.
- Future PPO adapter matches deployment behavior and stored/recomputed movement-action log probabilities; synthetic update/resume contracts pass. A changed P/resolver creates a new run identity.
- Search precision adapters, fixed-work replay and deadline/fallback counters are usable; tests do not label development timing as host evidence.
- Exact model/config hashes and new variant names, required core and relevant P-only tests, and a PR covering the declared lanes. No old checkpoint, artifact, frozen schedule or production default overwritten.
- Host validation/adaptation follow-ups specify exactly eight workers, 2s/50ms full-turn limits, fresh seeds, paired controls and fixed budgets before execution. This task closes on reviewed code/spec delivery, not an invented strength gain or an unperformed training run.
- Acceptance is a correct, reproducible new carrier and tested learning interface. Establish new-carrier baselines before formal fresh training/collection. Historical bot scores are comparison controls, not inherited strength guarantees; compatible old checkpoints are optional initialization.

## Handoff

*2026-10-16 01:47Z claude-review@caspers-macbook:*

**Done.** PR https://github.com/Casper015/ctf-fusion/pull/60 is on branch claude-review/t0065-rewrite-modular-cpu-carrier-around. It holds the redesigned agent_v2 carrier, its contracts and tests, and docs/AGENT_V2.md.

**Next:**
- Review and merge PR 60, then mark T0065 done.
- The host baseline is T0069, which depends on T0065.
- T0066 and T0068 wait on this interface. Fresh collection or training on agent_v2 should follow T0069's baseline.

**Not verified:** Windows/AVX2, timing (including first-reply startup with all P models), torch updates, strength.

**Not done:** Codex's final review was stopped at Casper's request; the closing review was mine (fixes in 8c16424).

## Log

- 2026-10-15 23:03Z codex-architecture@caspers-macbook: created
- 2026-10-15 23:04Z codex-architecture@caspers-macbook: handoff prepared
- 2026-10-16 00:08Z claude-review@caspers-macbook: claimed
- 2026-10-16 00:25Z codex-architecture@caspers-macbook: updated at Casper's explicit request; redesign permits changed behavior; preserved task owner/status/branch; gated fresh collection on T0065.
- 2026-10-16 00:27Z codex-architecture@caspers-macbook: Casper explicitly reiterated redesign, not refactoring; made it mandatory in the title, Goal and Handoff; owner/status/branch preserved.
- 2026-10-16 01:47Z claude-review@caspers-macbook: note: PR https://github.com/Casper015/ctf-fusion/pull/60 (rebased on c7dbfb6). Redesign per Casper: one public state, pure P, frozen meow schemas via versioned encoder, new collision-aware resolver assign_v1, fusion::step + lawful refreshes, default order before search, PPO bridge/trainer on the same path. Core 426 OK, P-only 64 OK, agent_v2 21 OK (2 torch skips). Actor-direct is the production default per T0064.
- 2026-10-16 01:47Z claude-review@caspers-macbook: handoff -> review
