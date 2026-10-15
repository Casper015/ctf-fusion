# Meowmind RL CPU bots: export, control, search arm and screen (T0056)

This turns a frozen actor from [MEOWMIND_RL_IMPLEMENTATION.md](MEOWMIND_RL_IMPLEMENTATION.md) (T0054's trainer, T0055's runs) into single-file CPU bots for the arena. It also builds the matched control and the switchable value-search arm, and specifies the deployment screens of [MEOWMIND_RL_PLAN.md](MEOWMIND_RL_PLAN.md) section 7 (`profiles/meow_rl_screen.json`).

**Status.** The pipeline, the control and the search arm are built and tested on the Mac with the warm-start actor and synthetic critics. **No trained candidate exists yet.** T0055's final checkpoints are pending, so no candidate bot, no host timing and no game result is claimed here. T0056 was started before T0055 on Casper's instruction (2026-10-15; the task log records it).

## The bot

| File | What |
|---|---|
| [bots/meow_rl/bot.hpp](../bots/meow_rl/bot.hpp) | The executable: the unmodified Meowmind reference with its search replaced by an actor-direct choice. It shares `meow_rl::Session` ([tools/cpp/meow_rl/session.hpp](../tools/cpp/meow_rl/session.hpp)) with the trainer's bridge, so features, mask, opponent belief, resolver and reply format are one implementation. |
| [bots/meow_rl/sample.hpp](../bots/meow_rl/sample.hpp), [sampling.py](../bots/meow_rl/sampling.py) | The deployed choice and its Python mirror. |
| [bots/meow_rl/search.hpp](../bots/meow_rl/search.hpp) | The bounded value search (`MEOW_RL_SEARCH=1`). |
| [bots/meow_rl/export.py](../bots/meow_rl/export.py) | Checkpoint → `inference/meow_rl/<name>/{actor.hpp, critic.hpp, MANIFEST.json}` + `bots/meow_rl/variants/<name>.cpp`. |
| [bots/meow_rl/variants.json](../bots/meow_rl/variants.json) | Every screen arm, for `python bots/p_only/tools/build.py --variants bots/meow_rl/variants.json` (bundled single-file bots, manifest of exact hashes; `run_screen.py` and `screen_runner.py` use them unchanged). |

Per turn, the bot:
1. reads the block and runs `Session::observe` (exactly the trainer's bridge);
2. chooses the joint movement;
3. runs `Session::resolve`.

A block the session refuses (malformed input) is answered with every unit staying and waiting, and is counted. stderr carries the model identity on turn 1 and the counters (`MEOWRL turns … refused … search_used … search_fallback … leaves …`) on the last turn.

**Deployed choice: sampled actor-direct.** PPO optimizes the sampled policy, so the predeclared deployment draws from the integer actor's softmax over feasible joints at T = 1.05. The draw is one inverse-CDF uniform.

The uniform for turn t depends only on the public INIT (teams, seat, size, turns, map, bases, spots) and t, never on the model. Control and candidate therefore face identical draws on a fixture, and an A/A copy replays identical choices. `MEOW_RL_GREEDY=1` builds the greedy diagnostic (ties to the lowest index).

**Integer format.** Actor and critic use the repository's INT16 format (reference16, `training/meow_rl/quant.py`, `inference/mlp.hpp`), Casper's decision. The headers carry the scheme, the integer digest and the run identity digest. On the recorded games, the bot equals the Python pipeline on every reply (bridge → integer actor → resolver), in both sampled and greedy modes.

## The matched control

`python bots/meow_rl/export.py --warm-start --name warm_start` exports the reference actor, the starting point of PPO, through exactly the same path as a trained actor: same bot, features, resolver, opponent model, integer format and calibration. The control and candidate bundles differ only in their actor header, so the primary contrast isolates RL.

The committed control `inference/meow_rl/warm_start/` agrees with its FP32 actor on 97.9% of greedy joints on the held-out recorded games (the format's precision limit, [MEOWMIND_RL_IMPLEMENTATION.md](MEOWMIND_RL_IMPLEMENTATION.md#integer-cpu-export)).

**The export is the same on every platform.** The calibrated activation ranges are computed in float64 and rounded up to 8 significant bits (`quant.range_of`). Last-bit differences in the statistics, such as BLAS summation order or compiler FMA in the features, therefore never change the integers. The test regenerates the committed control and requires the identical digest on every CI platform.

## Exporting a trained candidate (after T0055)

```bash
python bots/meow_rl/export.py --checkpoint runs/meow_rl/large_v_s101/checkpoints/final.pt --name large_v_s101 --calibration-batches runs/meow_rl/large_v_s101/batches
```

- **Inputs.** It needs PyTorch to read the `.pt` file (the host has it). `--actor-npz/--critic-npz/--identity-json` take NumPy arrays instead.
- **Refusals.** The export refuses:
  - a checkpoint whose identity differs from this code: actor and critic architecture, feature schemas, temperature 1.05, reward units, resolver, reference sha256;
  - a checkpoint whose NumPy actor snapshot differs from its PyTorch actor.
- **Calibration** uses the run's saved features (`--keep-features`). T0055's runs did not keep them. For those, `bots/meow_rl/calibrate.py` plays a few fresh games with the frozen actor through the trainer's own collector, on a seed reserved for calibration, and writes the same `update_*.npz` layout with a receipt:

  ```bash
  python bots/meow_rl/calibrate.py --checkpoint runs/meow_rl/large_v_s101/checkpoints/final.pt --pool profiles/meow_rl_training_pool.json --seed-base <reserved> --seed-ledger docs/MEOWMIND_RL_SEEDS.md --episodes 40 --workers 6 --out data/meow_rl_calibration/large_v_s101
  ```

  Then pass `--calibration-batches data/meow_rl_calibration/large_v_s101` to the export. Without any of this, the recorded 4-team game is used, which is fine for plumbing but too narrow for a trained export.
- **Manifest.** It records the agreement with FP32 on the held-out 2- and 15-team recorded games: greedy agreement, log-probability error, saturations and the critic's value error. A low agreement is a reason to fine-tune the actor against the format (the repository's QAT path) as a separate, recorded step before screening.
- Then add the new `meow_rl_<name>` entry to `variants.json`.

## The search arm

`--critic --search` builds the same actor with the bounded value search (`search.hpp`). Two-ply means two complete simultaneous turns, the reference's own deep-search shape:
- **Roots.** The actor's own choice, then the next best feasible joints by the actor's score, `MEOW_RL_SEARCH_CANDIDATES` (4) in all.
- **Scenarios.** `MEOW_RL_SEARCH_SAMPLES` (2) stochastic opponent order sets from the reference's `enemyOrders`, shared by all roots.
- **Ply 1.** `Planner::step`.
- **Ply 2.** Forecasts are recomputed at the successor, our continuation is one actor draw under the successor's mask, and the opponents' orders are drawn there; then `Planner::step` again.
- **Value.** `r1 + r2 + V(leaf)` with gamma 1: 0 except the exact terminal 2U − 1, counted once and never bootstrapped. V is the trained critic in the same units. The reference's own reward, value, potential and 0.97 discount are not used.
- **State.** The movement history, new-life flags and prepared turn are saved and restored around each expansion, as in the reference. Nothing is cached across branches.
- **Deadline.** `MEOW_RL_SEARCH_DEADLINE_MS` (30, a provisional value T0057's preflight freezes) is checked before each trajectory. An incomplete search is discarded and the actor's choice is played. Tested: with a zero deadline the search bot is byte-identical to the actor-direct bot on the recorded games.

**Cost (not measured here).**
- The large V is 1,999,360 multiply-adds per leaf; the default search evaluates at most 8 leaves and 16 transitions per turn.
- Its INT16 weights are about 4 MB, and its header source about 12 MB (the arena allows any source size).
- The bot needs `FUSION_MAX_WIDTH=1024`, which the generated variant defines.

On the Mac, with a 28,801-parameter stand-in critic, the 15-team recorded game averaged about 16 ms per turn and fell back on 4 of 400 turns. That is a development machine with a smaller critic: only the host's preflight measures the real arm. If the large V does not fit the 50 ms budget, the alternatives are separately named arms: fewer leaves, the small-V critic, or a distilled evaluator.

## The screens (`profiles/meow_rl_screen.json`, run by T0057)

Two separately frozen schedules, each a `run_screen.py` config once T0057 fills a fresh seed (reserved in `docs/MEOWMIND_RL_SCREEN_SEEDS.md` before the first game) and its frozen worker count.

**Opponents are held out (plan section 5; the split is docs/MEOWMIND_RL_POOL_SPLIT.md, T0058).** T0055 trained against `profiles/meow_rl_training_pool.json`, which holds 8 of the 12 bots of the default screen roster (`roster.strength.REFERENCE`). Training pool v2 (T0058) adds the AI_Test lineages:
- the Astra variants;
- the Player5 lineage: Player2_v005, Control_Hybrid/Omega, the C-series/R3 bots and the S1/S2 controls;
- Player1 and Player3.

Both screens therefore draw their opponents (`reference`) only from held-out lineages listed in `profiles/pool.json`: Fable, MiMo, Kimi, GLM and Player4. The test checks that the roster is disjoint from both training pools. Meowmind and Meowmind + P3-distilled remain secondary *arms* (they take our seat and are compared), never opponents.

| Schedule | Fixtures | Arms | Games | Endpoint |
|---|---:|---|---:|---|
| Primary | 392 (28 × counts 2–15) | L = `meow_rl_large_v_s101`, W = `meow_rl_warm_start`, L~aa | 1,176 | L − W, equal-count paired utility, 95% map-cluster interval, margin 0 |
| Secondary | 392 fresh | L, `meowmind`, `v12_ponly_xxl` (P2 production), `meowmind_p3_distill_xxl`, `meowmind_p3_xxl`, S = `meow_rl_small_v_s101`, Q = `meow_rl_large_v_s101_search`, Q~aa | 3,136 | six contrasts, Bonferroni (z = 2.638 each) |

The secondary contrasts:

| Contrast | Question |
|---|---|
| L − `meowmind` | Is the RL bot better than unchanged Meowmind? |
| L − `v12_ponly_xxl` | ... than the P2 production bot? |
| L − `meowmind_p3_distill_xxl` | ... than Meowmind with the distilled P3? |
| L − S | Does the large critic train a better actor than the small one? |
| Q − L | Does the value search add to the actor? |
| `meowmind_p3_distill_xxl` − `meowmind_p3_xxl` | Does distillation from the 8XL teacher help in games (same carrier)? |

Whole-bot comparisons across carriers answer product questions, not mechanisms. The analyzer (`bots/p_only/tools/analyze.py`) reports 95% intervals; the Bonferroni intervals are computed from its estimate and standard error. A missing arm makes its comparisons "incomplete", never zero-filled.

## Screen results (T0057) and their corrections (T0060)

T0057 ran both screens on desktop-g60ojud (`reports/host/meow_rl_screen/`): 1,176 primary and 3,136 secondary games, seeds 20261951 and 20261952, against the held-out roster, 6 workers.

- **Primary.** `meow_rl_large_v_s101 − meow_rl_warm_start` = **+0.0318 [+0.0080, +0.0557]: helps.** RL improved the actor over its own warm start against opponents it never trained on, mostly at 11–15 teams (+0.048). A/A: 392 of 392 identical score vectors.
- **Secondary, under its predeclared Bonferroni rule** (`reports/host/meow_rl_screen/secondary/MULTIPLICITY.md`, generated from the committed `screen.json`). T0057's `SCREEN.md` shows 95% intervals, because `analyze.py` then had no multiplicity option.

  | Contrast | Mean | Verdict |
  |---|---:|---|
  | RL − unchanged Meowmind | −0.030 | hurts |
  | RL − P2 production (`v12_ponly_xxl`) | −0.030 | **not shown** (95% said hurts) |
  | RL − Meowmind + P3 distilled | −0.068 | hurts |
  | large V − small V | +0.033 | helps |
  | search arm − actor-direct | −0.148 | hurts |
  | P3 distilled − P3 XXL | −0.005 | not shown |

**What T0060 fixed for the next screen:**
- `analyze.py --multiplicity bonferroni`, and `run_screen.py` forwards a config's `multiplicity` (and `margin`). A comparison of an arm that is not analyzed is now an error, not a silent skip.
- The bot prints its cumulative `MEOWRL` counters every turn, before the reply. T0057's counters were lost, because the runner stops the bot right after its last reply. `analyze.py` reports them with their coverage.
- The preflight fails a level on missing counters and on any seat's fault, records per-game details, timestamps and the arms' bundle and binary hashes, and reads the search deadline from the bundled source.
- `roster.strength` rows carry opponent faults, late replies and deaths, which `analyze.py` totals per arm.

The search arm's loss remains unexplained by telemetry, because T0057 has no counters. A likely contributor: the INT16 large critic's mean absolute value error is 0.21 on the −1…+1 scale (`inference/meow_rl/large_v_s101_search/MANIFEST.json`).

## Should we train a P2 8XL or 10XL teacher and distill it to XXL?

Casper's question, 2026-10-15. What the repository shows:

| Evidence | Prediction | Games |
|---|---|---|
| T0011 ([reports/models/p_features/RESULTS.md](../reports/models/p_features/RESULTS.md)): P3 vs P2 at XXL | joint NLL −0.088 (test, all three seeds) | — |
| T0013 ([bots/p_only/report/p_features/RESULTS.md](../bots/p_only/report/p_features/RESULTS.md)): P3 vs P2, 1,960 host games | — | v12 +0.011 [−0.018, +0.040], c08 +0.002 [−0.026, +0.030]: **not shown** |
| T0032 ([reports/models/p_distill/RESULTS.md](../reports/models/p_distill/RESULTS.md)): P3 8XL teacher → XXL | student −0.0116 validation loss (42% of the teacher's gap), all in the movement head; mixed4_8 QAT then +0.033 | never screened on the host |

P3's prediction gain over P2 was about seven times what distillation achieved, and no game effect was shown. A distilled P2 student would gain less than P3 did, so a screen of this size is unlikely to detect it. A teacher run also costs GPU days, and the 12 GB laptop GPU has not been shown to hold 8XL; T0032 trained it on an A100.

The two reviewers differ on what follows:
- **Claude:** do not start a P2 teacher now.
- **Codex:** a bounded P2 8XL trial (one teacher, matched supervised and distilled students, gated on the gain surviving the actual reference16 export) is reasonable because P2's features are cheaper. 10XL has no supporting evidence.

Both agree the secondary contrast `meowmind_p3_distill_xxl − meowmind_p3_xxl` should come first. It tests distillation in games at no extra training cost. If it is not shown, a larger P2 teacher has no game evidence behind it.

## Verification (caspers-macbook, clang 17 / g++-16, no PyTorch)

`tests/test_meow_rl_export.py` (11 tests) covers:
- **Parity:** bot = Python pipeline on every reply of the recorded 2/4/15-team games, sampled and greedy.
- **Uniforms:** they depend only on the public INIT.
- **Robustness:** a malformed block gets the safe reply and the game continues; end of input mid-block stops cleanly.
- **Export:** reproducible digests; identity refusals for temperature, features, reward units, resolver, reference sha and unknown critic.
- **Critics:** the full 416-1024-1024-512-1 critic fits the INT16 format with proved int32 bounds; the critic's C++ accumulators equal NumPy exactly.
- **Search:** with no time it plays exactly the actor; its choices are feasible and resolved by the reference resolver.
- **Spec:** variants file and screen spec are consistent (arms exist, game counts, the Bonferroni z).

All five bots in `variants.json` build with `build.py`.

**Not verified here:**
- any trained actor or critic, and reading a real `.pt` checkpoint (needs PyTorch);
- Windows builds;
- host timing of the actor-direct bot and the search arm;
- the 12 MB critic header's compile time on the host;
- every game result.
