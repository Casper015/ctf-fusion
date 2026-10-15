---
id: T0065
title: "Rewrite modular CPU carrier around existing P"
status: open
owner: ""
needs: []
writes: [bots/agent_v2/, inference/agent_v2/, training/agent_v2/, tools/cpp/agent_v2/, tests/test_agent_v2.py, tests/test_agent_v2_training.py, profiles/agent_v2.json, docs/AGENT_V2.md]
depends: [T0061]
land: pr
branch: ""
updated: 2026-10-15
---

## Goal

Build a simple, modular CPU carrier that reuses ctf-fusion's learned opponent P, existing PPO actors/critics, mathematical tactical assignment and measured inference kernels. This is a user-requested architecture rewrite for future learning experiments, not a CNN project. First deliver tested code and a migration contract; host training and strength runs follow separate frozen tasks.

### Intent and current evidence

Casper requested this task on 2026-10-15 (America/Los_Angeles). Retain the mathematical layer and the MLP direction. Replace the original Meowmind opponent predictor and its hand-written online expert/habit updater in the NEW carrier with our existing P. Preserve the original source, existing bots, checkpoint artifacts and completed evidence as controls.

Planning snapshot: main 58de8cf, PR #58 merged; T0062 is producing aligned exports and T0063/T0064 will diagnose and screen them. Read their latest status before working. This task is not a replacement for those experiments, does not alter their frozen inputs and must not borrow their test seeds. Code work can start independently; consume their conclusions before selecting a critic backend or enabling new search for deployment.

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
| `inference/policy.hpp`, `policy_p3.hpp`, their feature encoders and pinned P2/P3 model registries | Reuse through an explicit adapter. P2 XXL is the initial reference adapter; existing P3 and distilled P3 remain named alternatives, not assumed winners. No new P training here. |
| P's joint 125-move distribution, move marginals, action and target heads | Use marginals for the resolver and joint/conditional distributions for enemy scenarios. Retain squad coordination instead of independently sampling three movement marginals. |
| `training/meow_rl/` PPO, GAE, actors/critics, storage, pool/seed handling | Reuse tested logic. Add a new environment/bridge adapter and run identity without silently changing an old run's code or resume contract. |
| Warm-start, T0055 and T0059 actor/V checkpoints | Preserve and load only with exact compatible features, units and hashes. They are initialization/reference artifacts after a behavior change, not automatically admitted policies. |
| Meowmind geometry, prepared-state conventions, 112 actor features and 416 V features | Extract the required implementation with attribution and parity tests. Keep the schemas exact during the first migration. A shared public state does not mean forcing P and pi/V into one feature vector. |
| `Planner::shots` 8-subset DP | Extract into a resolver taking explicit state, moves and opponent move marginals. Keep the original coefficients and tie order for the first comparison. |
| Official rules and existing simulators | Retain authoritative semantics. Reuse or repair one simulation module after parity checks; do not write a new learned world model. |
| `inference/mlp.hpp`, `mlp_float.hpp`, `mlp_wide.hpp`; aligned export/calibration | Reuse numerical implementations and integrity checks. Choose production precision from evidence, not format names. |
| Search v2 common random numbers, staged budgets, fixed-tree harness and telemetry | Reuse as optional planning machinery, parameterized on the new predictor/resolver/state interfaces. It must never call the original predictor behind those interfaces. |
| Existing screen, all-seat fault accounting, exact-eight-worker preflight and frozen schedules | Reuse tooling; register new variants and fresh schedules for this carrier. Do not mutate T0062-T0064 inputs. |
| Original Meowmind learned enemy experts, heuristic expert mixture, evidence arrays and hand-tuned habit updates | Remove from the new production carrier once P replacement is complete. Allow the old implementation only in isolated migration tests/reference builds. |
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

The resolver consumes a joint move and opponent movement marginals, then returns complete own-team actions. Retain range/line-of-sight calculations, effective-hit/kill terms, pickup opportunity costs, subset assignment, tie-breaking and eligibility exactly for parity. Move fixed coefficients into one named configuration whose default equals the original constants.

Be precise about current limits: the resolver normally returns attack targets or pickup/wait; it does not currently generate voluntary drops. Merely exposing the X token is not a learned drop policy. Preserve that behavior initially. Do not conflate existing self-collision avoidance with an engine rule; the actor's feasible mask is a policy restriction and must be documented as such.

Complete candidate orders, not just movement keys, are passed to the simulator. Preserve exact flag semantics: attacks happen before pickup, pickup before voluntary drop, and drop before capture. A carrier cannot drop and fire in one turn, and a newly voluntarily dropped flag cannot be picked up in that same pickup phase. Cover these with engine-based fixtures before adding any later drop experiment.

### Flag randomness and simulation contract

The official environment has N+1 designated refresh spots and max(2, ceil(N/2)) flags. Free-spot selection excludes spots occupied by home flags; dropped flags remain on their cells. Return and cooldown timers are 15 and 8 turns. Verify all details against the engine pinned by the run rather than assuming versions never change.

Map seed, flag seed and actor sampling seed have distinct roles. The live actor never receives hidden flag RNG state or future flag spawns from logs. Search samples lawful refresh outcomes and handles multiple refreshes in engine order. Its RNG is a scenario generator, not the actual hidden engine RNG. Common random seeds across roots reduce variance but do not guarantee identical refresh events when candidate actions change RNG consumption or free spots.

Reuse map geometry and deterministic rule calculations. Check simulated transitions against the real engine, isolating stochastic refresh cases from deterministic parity. Cover all team counts 2-15, terminal timing, respawn immunity, movement collisions, attacks, pickup/drop/capture and return/cooldown order.

### Implementation sequence and attribution of changes

1. Inventory exact source/model paths and licenses/attribution; write `docs/AGENT_V2.md` with an architecture diagram and concrete interface contracts. Freeze migration fixtures and model identities.
2. Extract shared state/features, actor and resolver into the new runtime. A test-only old-predictor adapter must reproduce the legacy actor-direct behavior on full recorded prefixes. Preserve numerical and tie-order equivalence; explain every mismatch before proceeding.
3. Replace that adapter with our pinned P2 reference. First compare P outputs against its existing implementation on identical states/history, then validate full orders. Wire P action/target conditionals into optional enemy scenario generation. Keep actor weights and resolver coefficients fixed; this is the first behavior-changing arm.
4. Connect optional V/search through the same state and prediction interfaces. Use fixed work before equal-time gates. Keep branch state and randomness explicit. Preserve a complete actor-direct order before expensive search; record partial-work refusal and fallback.
5. Add an environment/bridge adapter for future PPO on the new carrier. It must execute the same P/resolver as the native deployment path. Reuse existing GAE/PPO/checkpoint code and build identity checks. Validate finite synthetic updates and log-probability recomputation without launching a full training run.
6. Deliver the single-file bundler/export configuration and a concrete host handoff with new variant names, commands, pinned models and required evaluation contrasts. Create dependent host validation/adaptation tasks through task.py when the code and relevant T0063/T0064 evidence are available; do not silently start matches or training as part of this code task.

No production source may include the entire original Meowmind .cpp with main renamed after extraction is complete. Preserve only the required, attributed implementations plus separately retained legacy fixtures/adapters. Reuse existing inference/trainer modules rather than copying whole parallel stacks.

### Subsequent learning, specified here but not bundled into the rewrite

- Compare legacy-predictor parity carrier versus our-P carrier with frozen actor/V and identical resolver settings. Both actor-direct and search need measured CPU feasibility; do not assume our P is cheaper.
- Once an opponent P is selected from evidence, freeze it and collect fresh PPO trajectories on the new runtime. Old actor weights can warm-start; old V weights can initialize only if feature/reward schemas match, and must be revalidated. Do not resume an old run while changing the environment silently.
- A small zero-initialized joint movement residual is the first optional actor extension: sum of existing per-unit scores plus delta(state, joint move). This is a separate ablation with its own identity, not part of structural parity. Reuse the concept of the P squad head, not opponent P weights as an own actor without training.
- Resolver coefficient selection and a drop/keep head remain conditional follow-ups after real-engine tactical diagnostics show headroom. A few discrete profiles are sufficient for a first experiment; Gaussian coefficients and full autoregressive actions are not required.
- CNNs and new teacher sweeps remain out of scope.

### Resource and evidence requirements

Final deployment is pure CPU, first turn 2 seconds and all later full replies 50 ms. Allocate immutable geometry/model buffers during initialization when useful; thread-local/per-branch scratch must remain isolated. Memory availability does not make stale cache reuse correct or guarantee 2-second loading. Include every served count/model in startup cost.

Every future official host match/preflight uses exactly EIGHT game workers under the CPU lease, with GPU training stopped. Freeze variants and settings before results; failure at eight stays failure, never silently reduce to six. Future CUDA training uses its own task and leases. No host timing claims from development machines.

Keep old evidence, raw logs and checkpoint backups. Shared calibration, sampling policy, model-family identities, data lineage, all-seat failures, search counters, actual work, paired map-cluster intervals and multiplicity stay explicit. Unchanged test predictions establish parity, not strength. Changed predictions establish neither improvement nor harm without matched evaluation.


## Done when

- A source-backed reuse/extract/remove inventory and architecture diagram in docs/AGENT_V2.md, including P versus own actor roles, legacy reference scope and the next learning tasks.
- A new single-file CPU carrier built from modular attributed code; actor-direct and optional search share one predictor/resolver path. No production dependency on the original expert mixture/habit updater or renamed-main monolith.
- Reused P model/encoder parity, actor/value feature parity, migration behavior parity under the legacy test adapter, deterministic mathematical resolver parity and complete-action protocol tests.
- Explicit causal-history, branch-isolation, target-slot, probabilities/marginals, cache identity and state-timing tests; simulator parity with the pinned official engine and lawful hidden flag randomness.
- Future PPO adapter matches deployment behavior and stored/recomputed movement-action log probabilities; synthetic update/resume contracts pass. A changed P/resolver creates a new run identity.
- Search precision adapters, fixed-work replay and deadline/fallback counters are usable; tests do not label development timing as host evidence.
- Exact model/config hashes and new variant names, required core and relevant P-only tests, and a PR covering the declared lanes. No old checkpoint, artifact, frozen schedule or production default overwritten.
- Host validation/adaptation follow-ups specify exactly eight workers, 2s/50ms full-turn limits, fresh seeds, paired controls and fixed budgets before execution. This task closes on reviewed code/spec delivery, not an invented strength gain or an unperformed training run.


## Handoff

*2026-10-15 23:04Z codex-architecture@caspers-macbook:*

Planning and source audit completed; implementation has not started. Read this task in full, then claim T0065 and use task.py start to obtain an isolated worktree. Current P2/P3 inference already exposes joint movement, marginals, action types and attack-target slots; preserve those interfaces instead of building a new predictor. Start with the inventory and legacy parity extraction, then replace the predictor. T0062-T0064 remain independent and their frozen artifacts must not change. The new runtime code can start now; host selection/training is a subsequent task, informed by the aligned results. No CNN, full autoregressive rewrite, new teacher run, or match batch was requested for this code task.

## Log

- 2026-10-15 23:03Z codex-architecture@caspers-macbook: created
- 2026-10-15 23:04Z codex-architecture@caspers-macbook: handoff prepared
