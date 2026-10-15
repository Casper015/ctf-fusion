# Meowmind PPO trainer: implementation (T0054)

This implements the trainer that [MEOWMIND_RL_PLAN.md](MEOWMIND_RL_PLAN.md) specifies, with [profiles/meowmind_rl_pilot.json](../profiles/meowmind_rl_pilot.json): a native bridge to the Meowmind reference, the live episode adapter, the masked joint actor, the large and small critics, GAE/PPO, durable storage with resume, an integer CPU export and the host CLI.

**Status.** Code and tests only. No training run has happened, and nothing here is evidence about playing strength. The first run of the complete CLI with PyTorch, the real engine and opponent processes together is T0055's smoke on the Windows host. What has and has not been run is listed under [Verification](#verification).

## Where things are

| Path | What |
|---|---|
| [tools/cpp/meow_rl/bridge.cpp](../tools/cpp/meow_rl/bridge.cpp) | The unmodified reference ([bots/meowmind_p/meowmind-ctf-374-20261011.cpp](../bots/meowmind_p/meowmind-ctf-374-20261011.cpp), sha256 `cc7b730c…`) compiled as a shared library with a C ABI (`#define main` renames its `main`; nothing else changes). |
| [tools/cpp/meow_rl/actor_int.hpp](../tools/cpp/meow_rl/actor_int.hpp), [actor_int_check.cpp](../tools/cpp/meow_rl/actor_int_check.cpp) | The integer actor for a CPU bot, on `inference/mlp.hpp`, and its parity tool. |
| [training/meow_rl/](../training/meow_rl/__init__.py) | `identity` (ids, profile check, source hashes), `bridge` (ctypes), `policy` (NumPy actor), `nets` (PyTorch actor and critics), `gae`, `ppo`, `pool`, `env` (live adapter), `store`, `quant` (integer export), `train` (CLI). |
| [tests/test_meow_rl.py](../tests/test_meow_rl.py), [tests/test_meow_rl_bridge.py](../tests/test_meow_rl_bridge.py) | Contracts below. The bridge tests need a C++ compiler, the PyTorch tests need PyTorch, the live test needs `AI_TEST_ROOT`. |

## One turn of one episode

The engine is the arena's `engine.py` (`fusion.paths.load_engine`). The learner is not a process: it runs inside the rollout worker.

1. `game.encode(i)` for every seat, once.
2. The inputs go to the opponent processes (`fusion.runner.Bot`, the same limits and fault rules as every match). Their processes live for the whole episode, so their memory persists.
3. The learner's own input goes to the bridge, which runs the reference's `Planner::run` up to its search: `adapt` (the opponents' online belief, causal: it uses the previous turn's movement and events), `learning::prepared` (predicted respawns), new-life flags, `predict`. The bridge then returns the 3 × 5 × 112 move features (`policyMoves`), the 416 value features (`Geometry::value`), the joint mask and, for tests only, the reference actor's own probabilities.
4. The NumPy actor snapshot samples one joint movement from the masked softmax. The bridge's `resolve` runs the reference's frozen resolver (`Planner::shots` with the reference's defaults, as `Planner::rollout` calls it) for exactly that movement and formats the reply as `Planner::run` does. The adapter checks that the reply carries the sampled movement and turn: there is no repair.
5. Only then are the opponents' replies collected. They can never reach the learner's observation.
6. Every reply is decoded and the engine steps once.

A bridge error, an altered reply or an engine refusal of the learner's reply raises `LearnerFault`; an opponent that cannot start raises `EnvironmentFault`. Either way collection starts no further episode, the whole batch is discarded and the run stops for a fix. The bridge refuses a malformed block (seat, ids, flags, holders or cells out of range) before the reference indexes anything. Each decision stores the sampled movement and the resolver's three actions (`resolved`), so the complete executed order is in the batch files. An opponent that times out or crashes mid-game gets empty orders, as in every match, and is recorded in the episode's `faults` and `opponents`.

## The actor and its probabilities

Per own unit k and move m, the shared MLP 112 → 64 → 32 → 1 (plain ReLU) scores the reference features x[k, m]. A joint movement j = 25a + 5b + c (moves `SUDLR`) has

`z_j = (f(x[0,a]) + f(x[1,b]) + f(x[2,c])) / 1.05`,  `log π(j|o) = z_j − logsumexp over feasible h of z_h`.

The temperature 1.05 is `Planner::policy`'s. With the reference weights (the warm start), the per-unit softmax `softmax(f(x[k, ·]) / 1.05)` over feasible moves equals the reference actor's `Planner::policy` probabilities within 2e-5 on the recorded games. The joint policy then conditions on feasibility (no shared destination), so a unit's marginal under the joint softmax can differ sharply from that per-unit value where collisions bind: this is the plan's masked joint actor, not a property of the reference to preserve.

**The mask is the reference's explicit feasibility only:** a dead unit stays, a non-stay move into a wall is infeasible, and two live own units may not share a destination. All-stay is always feasible. The reference's search also dropped moves below probability 0.00005. That rule depends on the policy, so it is **not** in the mask: on the 1,200 recorded turns the greedy warm-start actor picks the reference's own top joint on 1,199. The matched warm-start control of T0056 must use this same contract.

The worker stores the sampled index, its log probability and the mask. The learner recomputes the log probabilities with PyTorch before every update and refuses the batch if any differs by more than 1e-4. Search never chooses a behavior action here.

## Critics, rewards, GAE and PPO

- **Critics.** Fresh MLPs on the 416 value features (`Geometry::value`: 80 base features plus the three own units' stay encodings), ReLU hidden layers and a linear output. Large: 1024-1024-512, 2,001,921 parameters (`meow_public416_mlp_1024_1024_512_v1`). Small: 64-32, 28,801 parameters. PyTorch default initialization after `torch.manual_seed(101)`.
- **Reward.** 0 until the end, then `2U − 1` with the pairwise utility U of `roster.strength.utility`. The engine's fixed-length game makes the last decision a true terminal (bootstrap 0). `gae.gae` also supports truncated segments, which bootstrap the next observation's value, for a future window-limited collector.
- **Old values** are computed on the device after collection, with the critic of that update, before it is trained.
- **PPO** (`training/meow_rl/ppo.py`): the plan's settings plus the following fixed choices.
  - Separate Adam optimizers: betas 0.9/0.999, eps 1e-5, no weight decay. Gradient norm is clipped at 0.5 per network.
  - Advantages are normalized once per batch (population std, eps 1e-8). Ratios and reductions are float32.
  - Minibatches are reshuffled each epoch from the checkpointed generator.
  - Approximate KL `mean(ρ − 1 − log ρ)` is measured before each actor step. Above 0.01, the actor takes no further step in that update; the critic finishes all four epochs. The trigger is recorded.
  - A non-finite loss or gradient raises an error.
- **Budget.** Learner decisions, one per turn, never all-seat replies. Each update freezes one actor snapshot, collects complete new episodes until at least `rollout_decisions` (8192) and includes every episode it started.

## Opponents, schedule and seeds

The pool file (schema `ctf-meow-rl-pool-1`; T0055 commits `profiles/meow_rl_training_pool.json`) lists bots with `name`, `family`, `path` (arena-relative or `@repo/`) and optionally a pinned `sha256`. A pinned bot whose file differs is refused.

Each opponent seat draws a family by its share (equal by default), then a member uniformly. A pool whose shares exceed its `family_cap` is refused, and the trainer refuses a pool whose `family_cap` is above the profile's `family_share_cap` (0.15) unless `--dev`; with equal shares at least 7 families are needed. The cap holds for the share of all opponent seats, not within a duel. There is no learner self-play.

Episode i of seed base S draws everything from sha256 derivations of (S, purpose, i): team count (uniform over 2–15), learner seat, map and flag seeds, opponents and the actor's sampling seed. Worker timing never changes the schedule.

`--seed-ledger` must list `--seed-base` before the run (T0055 reserves it in `docs/MEOWMIND_RL_SEEDS.md`). `--dev` allows an unledgered seed and marks the run as development plumbing.

## Storage and resume

A run directory holds `config.json`, an fsynced `journal.jsonl`, `batches/update_NNNNNN.{npz,json}`, `state.pt` and `checkpoints/`. Every file is written atomically.

- **Commit point.** An update counts as committed when `state.pt` is replaced. It holds models, optimizers, the minibatch generator, the PyTorch RNG, counters and the next episode index.
- **Restarts.** On `--resume`, episodes the journal shows as started but not committed are logged as discarded, and their indices (seeds) are never reused. Batch files of an uncommitted update move to `discarded/`. Opponent processes cannot be restored, so an interrupted collection is discarded, not continued.
- **Matching configuration.** A resumed run must match the stored configuration except mode, budget, device, workers and Python version. A changed source file under `training/meow_rl/` or `tools/cpp/meow_rl/`, a changed profile or pool, a changed opponent (its source sha256 and the sha256 of the executable it compiled to are both recorded) or a changed seed base is refused.
- **Checkpoints.** At the first committed update at or after every 100,000-decision mark, named by the actual count (`checkpoints/decisions_0000100800.pt`: updates hold whole episodes, so a mark is passed, not hit exactly); `smoke.pt` at the smoke budget, `final.pt` at the full budget. A checkpoint file is written before `state.pt`, so a committed checkpoint always exists. The final checkpoint is the predeclared candidate.
- **Robustness.** Recovery removes a torn last journal line (kept in `journal.torn`, reported) before anything is appended, and moves stray temporary files to `discarded/`.
- **Budget accounting.** The budget counts trained (committed) learner decisions. Executed but discarded work is counted separately and rebuilt from the journal on every resume: `discarded_episodes` (every started episode of a discarded batch or interrupted collection) and `discarded_decisions` (decisions of episodes that finished before their batch was discarded; an interrupted episode's own decisions are unknown). Both are in the counters and the smoke report; they never count toward the budget, which is trained decisions.

## Integer CPU export

`training/meow_rl/quant.py` exports the actor in the repository's **INT16 format**: `reference16` in [P_MIXED_PRECISION.md](P_MIXED_PRECISION.md), the format `training/export_weights.py` writes and [reports/host/SIMD_OPTIMIZATION_REPORT.md](../reports/host/SIMD_OPTIMIZATION_REPORT.md) benchmarks:
- int16 weights and int32 biases;
- int32 accumulators proved inside ±2^30;
- hidden activations on the 0..128 scale, power-of-two shifts and floor(x + ½) rounding;
- real output = acc / 2^out_exp.

The bot evaluates it with `inference/mlp.hpp`'s `fusion::forwardRaw` (`actor_int.hpp`). `fusion.quant.forward_raw` is its exact emulation; C++ and NumPy agree bit for bit (clang++ and g++ tested).

Two exact steps fit the plain-ReLU actor into that clipped-ReLU format:
1. **Inputs.** Features (clamped to [−4, 4]) become `floor(x·2^12 + ½)`.
2. **Activations.** ReLU is positively homogeneous, so each hidden neuron is divided by its calibrated range (1.1 × its largest activation on recorded observations) and the next layer's column is multiplied by it. The float function is unchanged except where an activation exceeds its range and saturates; `quant.agreement` counts those cases.

The output scale is shared, so the greedy joint compares int64 sums of three accumulators (ties to the lowest index).

**Precision, measured on the warm-start actor** (calibrated on the recorded 4-team game, checked on the recorded 2- and 15-team games, 800 decisions):

| Format | Greedy joint = FP32 | Max \|Δ log π\| | Mean \|Δ log π\| |
|---|---:|---:|---:|
| reference16 (INT16, shipped) | 97.4% | 2.97 | 0.29 |
| int16 weights + int16 activations, int64 accumulators (not shipped) | 99.75% | 0.06 | — |
| int8 per-row weights + int16 activations (not shipped) | 93.3% | 7.5 | — |

The limit of reference16 here is its 129 activation levels. The reference actor's logits span about −41 to +30. Casper chose the existing format (2026-10-15); a later step (T0056) can fine-tune the trained actor against the format with the existing QAT path if its own agreement is too low. Collection never samples from the integer actor: it is a deployment artifact, checked against the FP32 snapshot by `quant.agreement` on recorded rollouts at export.

## Windows

- **Bridge DLL.** `training.meow_rl.bridge.build` compiles the bridge with `fusion.runner.find_compiler` (w64devkit `g++` on the host), using `-std=c++20 -O2 -shared -static`. The DLL carries its own libstdc++/libgcc/winpthread, is named by a content hash under `build/meow_rl/`, and is loaded by absolute path. Every entry point catches C++ exceptions. The reference parser reads `std::cin`, so a process runs one session at a time; rollout workers are processes.
- **Workers.** Spawned with the `spawn` context (as Windows requires). They never import PyTorch and never touch CUDA: the actor snapshot is a dict of NumPy arrays.
- **Opponents.** They run through `fusion.runner.Bot`, whose reader threads work on Windows pipes.

## Host commands (T0055)

Tested here only with a synthetic collector (CI, PyTorch on CPU) and, separately, with the real process collector on the public arena without PyTorch. Run inside the leases:

```bash
python tools/repo/task.py lock gpu --task T0055 -- python -m training.meow_rl.train --mode smoke --arm large_v --pool profiles/meow_rl_training_pool.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --device cuda --workers <n> --out runs/meow_rl/large_v_s101
```

```bash
python tools/repo/task.py lock gpu --task T0055 -- python -m training.meow_rl.train --mode full --resume --arm large_v --pool profiles/meow_rl_training_pool.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --device cuda --workers <n> --out runs/meow_rl/large_v_s101
```

The smoke writes `smoke_report.json`:
- non-zero actor and critic gradients;
- the largest log-probability recompute gap;
- finite losses;
- an exact round trip of the saved state.

The small-V control is the same pair of commands with `--arm small_v`, its own fresh seed base and its own `--out`. Collection limits default to the official 2 s / 50 ms for opponents; record a different `--limits` if used.

## Verification

Run on caspers-macbook (Apple clang 17 and g++-16, Python 3.15, no PyTorch):

| Done-when item | Test | Run here |
|---|---|---|
| Feature parity | `Bridge.test_batched_features_equal_the_scalar_reference`, `test_carrier_hp_reaches_the_features` (enemy carrier HP 100 vs 32 → 1.0 vs 0.32) | yes |
| Actor parity | `Bridge.test_warm_start_actor_matches_the_reference` (≤ 2e-5 on 1,200 turns, 2/4/15 teams) | yes |
| Resolver parity | `Bridge.test_resolver_and_reply_equal_the_original_bot`: the original compiled without search, every reply identical on 1,200 turns | yes |
| Causal observations | `Bridge.test_observations_do_not_depend_on_the_learners_choices`, `Adapter.test_learner_acts_before_opponent_replies…` (send → observe → receive → step every turn) | yes |
| Sampled log probabilities | `JointPolicy.*` (125-way oracle, sampling), `Adapter…` (stored = recomputed), `TorchLearner.test_numpy_and_torch_actors_agree_and_the_first_ratio_is_one` | NumPy yes; PyTorch in CI |
| Legal masks | `Bridge.test_masks`, `JointPolicy.test_a_very_unlikely_feasible_move_keeps_positive_probability`, `TorchLearner.test_an_action_outside_its_mask_is_refused` | NumPy yes; PyTorch in CI |
| Terminal/truncation GAE, tied utility 2–15 | `Rewards.*` | yes |
| Finite updates, KL stop | `TorchLearner.test_update_is_finite_and_moves_both_networks`, `test_kl_stop_freezes_the_actor_but_not_the_critic`, `test_critic_sizes` | CI |
| Checkpoint resume | `TorchLearner.test_resume_reproduces_the_next_update_exactly`, `Storage.*`, `Cli.test_an_interrupted_collection_is_discarded_on_resume`, `Cli.test_smoke_then_full_resume` | storage yes; PyTorch in CI |
| Source/config identity | `Identities.*`, `Cli.test_resume_refuses_a_different_run_and_requires_a_ledger` | partly; PyTorch in CI |
| Protocol/faults | `Bridge.test_protocol_errors`, `Adapter.test_a_learner_reply_the_engine_refuses_is_a_fault`, `test_an_opponent_that_cannot_start…` | yes |
| Live engine | `LiveEngine.test_one_live_episode` (public meow-ai-arena, 3 teams, Baseline) | yes, plumbing only |
| Integer export | `Bridge.test_integer_actor_is_bit_exact_in_cpp`, `Export.*` | yes |

Also run by hand: three spawned workers (`ProcessCollector`) played 2- and 3-team episodes against Baseline and the unmodified Meowmind on the public arena. Every episode had 400 decisions, real scores and no faults, and the learner took 0.3–0.5 ms per decision. This is a development machine: none of it is timing or strength evidence.

**Not run here:**
- PyTorch code (CI's CPU-torch job runs `tests.test_meow_rl`);
- CUDA;
- the Windows DLL (CI's Windows leg builds it when a compiler is on its path; the host smoke is the real check);
- the complete CLI with the real collector;
- throughput at the host's worker count;
- opponent fault rates under load.
