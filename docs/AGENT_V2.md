# agent_v2: a CPU bot architecture designed around our opponent model P (T0065)

agent_v2 is an **architecture redesign, not a refactor** of the Meowmind carrier (Casper, 2026-10-15). Its modules, data flow and contracts are designed for the current goals:

- our own opponent model P;
- a mathematical tactical resolver with a declared rule;
- the PPO actor and critics through versioned input contracts;
- the engine's rules, with training and deployment running one decision path;
- the 2 s / 50 ms CPU limits.

Old code is reused only where it earns its place. Old bots remain external strength controls, not specifications: agent_v2 is expected to decide differently, and nothing tests it for equality with the Meowmind bot's replies, scores or tie order.

This task delivers code, contracts and tests. No game, preflight or training run has used it. Host baselines and learning follow as separate frozen tasks ([Next](#next-host-baseline-then-learning)).

## Architecture

```text
public observation (protocol text) + causal history (the previous REAL State)
                 |
       parse once (tools/cpp/stream.hpp) -> fusion::Static + raw fusion::State        core.hpp
       MapCache (fusion::Board, rules), built at INIT; makeView: respawn origins, immunity
            /                                   \
   own actor pi                              our opponent model P (ponly::View, unchanged)       predictor.hpp
   meow_policy112_v1 inputs (encoder)        raw State + previous real State, per opponent team:
   joint movement over its feasible mask     joint movement, marginals, P(act | move), P(target slot | attack, move)
   (one sampled draw, common uniform)              |                              \
            \                                      v                               v
             '--> resolver assign_v1 <-- P's scenario table (K complete opponent order sets)      resolver.hpp
                  engine movement + attacks per scenario (fusion::step), declared score, exact enumeration
                          |
                  COMPLETE DEFAULT ORDER (ready before any planning)                              runtime.hpp
                          |
            [optional bounded search: roots x P scenarios; fusion::step + lawful flag refreshes;   search.hpp
             successor: branch P (parent = real root), continuation, resolver on the branch's table;
             V(meow_public416_v1 leaf) via fp32 / reference16 / WideAct16; any refusal keeps the default]
                          |
             protocol reply (complete orders) -> History.advance (once per real turn)
```

The runtime has two entry points:
- **Bot:** `decide(block, started, why)` reads one block of text and returns the reply.
- **PPO bridge:** `observe_block(text)` and `answer(joint)`, the same parse, predictor, encoder, resolver, reply and history commit, with the trainer's sampled joint.

Modules are concrete and selected at compile time: the P registry tier, the actor header, the critic header and `NoSearch`/`Search`. There is no plugin framework and no virtual dispatch. The shared state does not mean shared weights: P, the actor and the critic each read it through their own versioned input contract.

## Reuse / extract / remove (source-backed)

| Asset | In agent_v2 | Why |
|---|---|---|
| `fusion::Static/State/Board/TurnView`, `makeView`, `canAttack` (`inference/state.hpp`); `tools/cpp/stream.hpp` parser | **The public state and its geometry**, unchanged | Engine-mirrored, with each rule citing the engine. It is P's training representation, and one parser serves every module. |
| `inference/sim.hpp` `fusion::step` | **The simulator**, unchanged, plus our lawful refresh sampler (`simulate.hpp`) | Engine phases 1–6 on the public state. The engine test shows it identical to the official engine. |
| `bots/p_only/include/ponly.hpp` `ponly::View`, `inference/policy.hpp`, `features.hpp`, P2/P3 registries | **Our opponent model P**, unchanged | The existing P with its own feature, normalization, history and target-slot contracts. P2 XXL (`models_pxxl_vm`) is the first reference candidate; P3 and distilled P3 remain named alternatives. Pure P, no α mixture. |
| Meowmind reference lines 20–36 (data structs) and 173–341 (`learning::prepared`, `Geometry::policyMoves/value`) | **The encoder of the frozen schemas** `meow_policy112_v1` / `meow_public416_v1`, extracted verbatim (`meowmind_features.hpp`, `extract.py --check`) | These are exactly the inputs the existing PPO actors and critics were trained on, so the original code computes them, from the shared State. A new actor or critic would get a new schema next to it. |
| `tools/cpp/meow_rl/actor_int.hpp`, aligned actor/critic headers (T0062), `inference/mlp*.hpp` kernels | Reused | The integer actor and the FP32 / reference16 / WideAct16 critics, with their existing bounds and parity checks. Production precision follows evidence (T0063/T0064). |
| Search v2 machinery (T0061) | Re-designed on these interfaces | Common random numbers, staged admission, complete fallback and telemetry. Every prediction comes from P. |
| `training/meow_rl` PPO, GAE, networks, storage, pools, seeds, collector, `env.run_episode(lib_path=)` | Reused unchanged through `training/agent_v2/train.py` | It substitutes three things: the bridge library, the warm-start actor and the run identity. |
| Meowmind parser, `World`, `Planner::shots` and tie order, expert mixture, evidence and habit updates, own-policy network, hand potential, beam search, renamed-main monolith | **Not used** | Replaced by the shared state, `fusion::step`, `assign_v1` and our P. The production bundles contain none of them (tested). |
| CNN or Transformer, new teacher, full autoregressive attack policy, offline own-action scorer | Excluded | Not part of this redesign. |

The Meowmind reference is another author's bot. Casper published it unmodified, and its source carries no license notice (`bots/meowmind_p/README.md`). The extracted file names its origin, line ranges and sha256 `cc7b730c…`; no license is claimed.

## Contracts and versions

| Version id | Meaning |
|---|---|
| `agent_v2_v1` | the carrier (`kCarrier`) |
| `agent_v2_actions_v1` | an action is the actor's sampled joint movement (125) plus `assign_v1`'s complete orders. These are a deterministic function of the state, that joint and the frozen scenario seeds. |
| `assign_v1` / profile `dp_v1` | the resolver rule and its tunable coefficients |
| `meow_policy112_v1`, `meow_public416_v1` | the actor and critic input schemas |
| registry tier + `MANIFEST.json` sha256 + per-header sha256 | our P |

**Our opponent model P.** It is an opponent model, not our actor.
- **Inputs:** the raw State (never a prepared one) plus the previous real State as history. Our own team is never predicted.
- **Outputs, per opponent team:**
  - normalized joint movement over 125 moves, with consistent unit marginals;
  - `P(wait, pickup, drop, attack | move)`, where the mask means "may have an effect", not protocol validity;
  - `P(target slot | attack, move)` over eight candidates (nearest first, within distance 4). A slot maps to a world unit id, or is empty; candidates beyond eight are truncated, so a missing slot means missing support.
- **Factorization:** teams are independent. Within a team the movement is coordinated, while action and target heads are per unit given its move.
- **Collisions:** P does not mask squad collisions.
- **Cache identity:** every prediction carries the content key of its state, its parent and its model. Serving it for any other state is refused (`Counters::inconsistent`; tested).
- **Lifetime:** the root prediction lives for one real turn. A branch prediction is built for one simulated state, with the real root state as parent, and dropped.

**History.** It is the previous real State only.
- It advances once, when a real turn's reply is committed.
- A branch never writes it, which is tested together with the root prediction and the default order.
- A refused block clears it.

**The actor.** π reads `meow_policy112_v1` (112 features per unit move, with each unit's previous real cell as its movement history).
- **Feasibility:** it samples one joint over its feasible mask: walls and dead units stay, and no two own units share a destination. That mask is a **policy restriction**, not an engine rule.
- **Sampling:** the uniform for turn t depends only on the public INIT.

**Resolver `assign_v1`** (declared in `inference/agent_v2/resolver.hpp`; tested against an unpruned brute force over every target tuple, and against engine rules).
- **Scenarios:** the scenario table is K = 8 complete opponent order sets, drawn from P's team joint and conditional heads.
- **Movement:** per scenario, `fusion::step` resolves every unit's cell after movement (shared destinations, opposing swaps, blocking chains) and our units' survival against the opponents' drawn attacks.
- **Score:** for each assignment where each able unit attacks one enemy or none:
  > Σ over targeted enemies, Σ over scenarios of p·(hit·min(h, need) + [h ≥ need]·(carrier? carrier_kill : kill)) − Σ over attackers of the forgone pickup value C_k.

  C_k is the scenario-weighted value of picking up an available flag on the unit's cell, if the unit survives. On our own base the pickup scores immediately.
- **Eligibility:**
  - attackers must be alive, not respawn-immune and not carrying;
  - targets must be living, non-immune enemies;
  - candidates are pruned soundly, to within Manhattan 3 of the unit's origin or its wall-clamped destination.
- **Ties:** fewer attackers, then the smallest target tuple.
- **Orders:** attackers attack, other non-carriers pick up, carriers wait. **assign_v1 never drops a flag.**
- **What it optimizes:** it is exact for this declared score, not for winning. `dp_v1` starts from the Meowmind coefficients (0.045, 0.20, 0.65, 1.2, 0.32), as settings rather than rules.

**Simulator.** `fusion::step`, with complete orders for every unit.
- **Refreshes:** a returning flag gets a lawful sampled spot, mirroring engine `free_spot`. Flags are processed in id order; each picks uniformly among the designated spots not under a home flag (earlier refreshes included), or among all spots if none is free.
- **Hidden RNG:** the real flag RNG is never known to the policy or the search.
- **Common seeds:** they reduce variance across root candidates, but do not guarantee identical refreshes.

**Search (optional, experimental).**
- **Structure:** roots (the actor's joint plus the next joints by accumulator) × kSamples P scenarios, two plies.
- **Successor:** a branch P prediction and engine view whose parent is the immediate raw parent (the real root), the actor's continuation with the root's raw cells as history, and the resolver on the branch's own table.
- **Value:** V from the selected critic backend.
- **Admission:** staged (setup, trajectory, expansion, critic, after the evaluation, plus a reply reserve). Any refusal keeps the complete default order, which is resolved before the search starts.
- **Fixed work:** `check tree` replays fixed-work trees for offline backend comparison.

**PPO.**
- **Bridge:** `tools/cpp/agent_v2/bridge.cpp` exports `training.meow_rl`'s ABI 1 over the same Runtime.
- **What is stored:** the trainer's actor samples a joint and the bridge executes exactly that joint. So PPO stores the probability of the executed movement joint; the rest of the action is the resolver's deterministic function of it.
- **No search in training:** the bridge never searches, so a search-selected joint is never a PPO sample.
- **Run identity:** `training.agent_v2.identity` keeps meow_rl's model contracts, so compatible weights can initialize a run. It adds the carrier, action schema, resolver, predictor and source digest, so a changed P or resolver is a new run and `--resume` across carriers is refused.

**Intentional differences from the Meowmind carrier.**
- Opponent forecasts come from our P, not the expert mixture or habit updates.
- Tactics come from a declared, collision-aware scenario score, not Meowmind's marginal DP and its tie order.
- The simulator is `fusion::step`, not Meowmind's World.
- The refresh rule mirrors the engine.
- A complete default order is ready before any planning.
- Malformed blocks are refused instead of read.

With the same actor, movement is unchanged on the recorded streams. Tactical replies differ on 59, 38 and 45 of 400 turns (2, 4, 15 teams). That is expected, and it is not evidence either way.

## Validation on this Mac (GCC 16, AArch64)

| Gate | Result |
|---|---|
| Frozen feature schemas from the shared state against the original encoder (`tools/cpp/meow_rl` bridge) | identical features, masks, unit legality and value features on all 1,200 turns of the 2-, 4- and 15-team streams |
| Resolver: pruned `assign_v1` against the unpruned brute force; engine-rule properties | identical on 17,578 cases (10,601 attacks); no carrier or immune attacker, no invalid target, no drop, every attack has positive score |
| P contracts | marginals sum to 1, slots are empty or enemies, own team never predicted, complete scenario tables, stale predictions refused |
| Simulator against the official engine (meow-ai-arena `ctf/workspace/engine.py`), random complete orders incl. drops and friendly fire | identical units, scores, flags and attacks on 10,050 steps (counts 2–15) with the engine's spots injected; every sampled refresh lawful |
| Branch isolation | root prediction, history and default order unchanged by search, about 3,100 branch predictions per stream |
| Training = deployment | bridge replies equal the deployed bot's on 1,200 turns; an episode's stored log-probabilities equal the recomputed probability of the executed joint; resolved orders are reproducible |
| Protocol | 7 tokens, turn number, `SUDLR`, targets are enemies, carriers wait, no `X`; malformed blocks refused with the safe reply and the next turn answered |
| Bundles | single files (32 MB actor-direct, 41 MB with search), no reference parser, World, Planner, experts or renamed main |

**Not verified here:**
- Windows/MinGW and AVX2;
- any timing (development-machine numbers are never evidence), and the 2,000 ms first reply with every served P model linked in;
- PyTorch updates (the synthetic smoke and resume tests skip without torch);
- strength.

## Variants

All variants use the T0059 pool-v2 large-V actor (aligned reference16 export) and P2 XXL, and every name is new:
- `agent_v2_p2xxl_direct`: actor-direct, **the production default**.
- `agent_v2_p2xxl_search_{w16,fp32}`: experimental search, one per critic backend.
- `agent_v2_p2xxl_search_r16`: diagnostic only.

T0064 ([VERDICTS.md](../reports/host/meow_rl_aligned/VERDICTS.md)) measured, on the meow_rl carrier:
- the aligned p2_large actor helps against its warm start (+3.84 points [+1.66, +6.01]);
- search hurts with every critic: FP32 −3.91, WideAct16 −4.66, reference16 −14.76 points against direct play;
- WideAct16 recovers +10.10 points over reference16.

So search stays an unscreened experiment on agent_v2 until that deficit is diagnosed, and reference16 is not a search candidate.

Build with `python bots/p_only/tools/build.py --variants bots/agent_v2/variants.json`. Pins and settings are in [`profiles/agent_v2.json`](../profiles/agent_v2.json).

## Next: host baseline, then learning

Each step is a separate frozen host task with exactly 8 workers, 2 s/50 ms full replies, fresh ledgered seeds, paired controls and fixed budgets.

1. **Host baseline.**
   - Build on Windows and run the test suites there.
   - Run an eight-worker preflight of the four variants: counts 2/5/8/15, 16 maps each, all-seat faults, counter coverage, first-reply startup.
   - Run a paired screen of `agent_v2_p2xxl_direct` against `meow_rl_aligned_p2_large`. This is a carrier comparison: several things change at once (P, resolver, simulator), so it is not a single-factor mechanism result.
2. **Search on agent_v2,** only after the search deficit T0064 measured is diagnosed: real-engine continuations, and fixed-work trees on agent_v2 (`check tree`). Then compare WideAct16 or FP32 against direct as equal-time packages, in one Bonferroni family.
3. **Fresh PPO on the frozen carrier** (`python -m training.agent_v2.train --init-actor <actor .npz> ...`).
   - New run identity and fresh data; never resume an old run.
   - Old actors can initialize; old critics only after revalidation.
   - T0066's formal per-count collection and T0068's specialists wait on this interface (T0065), and T0068 needs a compatible common initialization.
4. **Later ablations, each with its own identity:**
   - a zero-initialized joint movement residual;
   - resolver profiles;
   - a drop/keep head, only after tactical diagnostics show headroom, and with its own drop specification and tests.
