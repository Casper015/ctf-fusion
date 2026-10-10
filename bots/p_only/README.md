# P-only bots: the new version with no hand-written opponent prediction

This folder holds everything needed to build and test the new bots. They are the two carriers, C08 Player5_Omega_LIT and Player5 v12 (rollout_volley), with **every prediction of what opponents will do coming from the trained policy network P**. The hand-written forecasts, the habit and target guesses, and the scripted opponent moves in the lookahead and playouts are deleted, not just switched off. Our own planning is unchanged: task assignment, candidate moves, scoring, search and evaluation.

The host agent's instructions are in [TASK.md](TASK.md). Start there after the quick start below. The next controlled runs (capacity and sampling, then V at fixed P) are specified in [EXPERIMENTS.md](EXPERIMENTS.md).

## Quick start (host)

The commands run from the repository root, with the AI_Test checkout in `AI_TEST_ROOT` (or pass `--ai-test`), Python 3.10+, NumPy and a C++20 compiler (w64devkit on Windows). They need no edits.

```sh
python -m unittest discover -s bots/p_only/tests -v          # ~20 s: no hand-written identifiers, bundles, contract + engine comparison
python bots/p_only/tools/build.py --stats                     # ~2 min: 17 bundled single-file bots + build/p_only/manifest.json
python bots/p_only/tools/run_screen.py --smoke                # ~1 min: 10 games, relaxed limits, checks the chain (not evidence)
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_capacity.json   # stage 1 (EXPERIMENTS.md), 2,520 games
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_value.json      # stage 2 (EXPERIMENTS.md), 2,240 games
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/timing.json   # 105 games: timing under screen load
python bots/p_only/tools/size_rule.py --apply                 # admits the C08 and v12 sizes into screen.json
python bots/p_only/tools/run_screen.py                        # the paired screen (bots/p_only/profiles/screen.json), official limits
```

The screen writes `bots/p_only/report/screen/SCREEN.md`, `screen.json`, `RUN.json`, the schedule and the build manifest. Commit those; raw results and match logs stay in `data/p_only/screen/` (not committed). If the run is interrupted, re-run the same command: finished games are kept. The analysis exits with status 2 and marks the report INCOMPLETE while any arm misses a fixture. Do not delete `build/p_only` during a screen: result rows are matched to the binary hashes in its manifest.

The offline structure scan is already generated at [report/STRUCTURE_SCAN.md](report/STRUCTURE_SCAN.md). Re-run it with `python bots/p_only/tools/structure_scan.py`; it plays no games.

## What is in the folder

| Path | What it is |
|---|---|
| `include/ponly.hpp` | The P-only opponent model. A `View` holds the network's predictions for one state (the real root or a simulated state): joint team moves and unit marginals, P(act \| move), P(target \| attack, move), mode and seeded draws. With no model it falls back to uniform over what the rules allow, and counts every fallback. |
| `bots/c08_ponly.hpp`, `bots/v12_ponly.hpp` | The new carriers. Each header comment lists exactly what was removed and kept. |
| `bots/variants/*.cpp` | Thin wrappers: policy size S, M, L, XL or XXL, and the experimental value term (`_m_v`). |
| `profiles/variants.json` | All buildable bots: the P-only variants, the C08 controls (4-turn scripted as frozen, and `c08_control_2ply` with 2-turn playouts and nothing else changed), the v12 control, and the earlier mixture ("hybrid") integrations from `bots/hybrid/`. |
| `profiles/timing.json` | Phase 1 timing: team counts 8, 12 and 15, every C08 P-only size and v12 at XL and XXL, at the screen's worker count. |
| `profiles/screen.json` | The paired screen: arms (the C08 P-only size set by `size_rule.py --apply`), re-run (A/A) copies, comparisons, seed 20261031, 20 maps per count. |
| `tools/bundle.py` | Inlines a bot's local includes into one `.cpp`, the file a judge compiles and the identity a screen records. |
| `tools/build.py` | Bundles and compiles the variants and writes the manifest (bundle and binary SHA-256, every inlined file's SHA-256, compiler, flags). |
| `tools/run_screen.py` | build, `roster.strength` schedule and run (identical fixtures for every arm), then analyze. |
| `tools/analyze.py` | Paired differences per team-count band with 95% intervals, re-run noise, faults, late replies, worst-game p99 and max reply time, network calls per turn, and C08 expectimax samples per call. |
| `tools/size_rule.py` | The fixed Phase 1 rule: the largest C08 P-only size with worst-game p99 at most 40 ms and no fault or late reply; v12 at XL and XXL. `--apply` writes the screen's arms. |
| `tools/structure_scan.py` | Offline triage of all 258 eligible bots: opponent model, lookahead, sampling, time budget, habits, joined with the light screen. |
| `tools/record_stream.py` | Records the input a seat receives in a game, and every team's orders (the test fixtures); several bots rotate over the seats. |
| `tools/screen_runner.py` | Plays a screen fixture by fixture with the arm order rotated, binds every row to an identity digest (bundle, binary, inlined sources and models, runner, schedule, limits) and resumes only identical evidence. |
| `profiles/screen_capacity.json`, `profiles/screen_value.json` | The next two controlled stages, with fresh seeds (EXPERIMENTS.md). |
| `tests/replay_test.cpp` | Fixed-work replay: per turn, a digest of the network's root predictions and the reply (V-off equivalence, P unchanged by V). |
| `tools/derive/make_*.py` | How the P-only headers and the C08 control (the original plus a stats line compiled only under PONLY_STATS; without it, identical assembly) were derived from `bots/hybrid/*_Fusion.hpp`, with every splice checked. A test re-derives them and compares. |
| `tests/` | `test_p_only.py` (Python), `contract_test.cpp` (C++ contract), `fixtures/` (two recorded games: inputs and every team's orders). |
| `report/` | Generated results: the structure scan now, the screen later. |

## What the network now predicts, and where

| Place | C08 P-only | v12 P-only |
|---|---|---|
| Root forecast (every opponent unit, every legal move) | P marginal; P(attack), P(pickup), P(drop) after the move; P(attack and target = each of our units) | P marginal; P(attack); aim at each of our units |
| Incoming fire in our joint-move scoring | P's aim at our unit, counted only when the settled cells allow the shot. This replaces the priority, crossfire and attention weights and the rule that an enemy always shoots. | Same |
| Threat map (carrier routes, escorts) | P(move) × P(attack) over the cells each destination can hit | Same |
| Expectimax samples (ply 1) | One joint draw per team from P per sample, the same draw for every plan | (no expectimax) |
| Ply 2 of every rollout | Once per sample, the exact turn kernel (stepExact) settles turn 1 under the best plan so far: our plan's moves and acts, each rival's sampled move with P's act and target for it. Deaths, HP, pickups, drops, captures and attack events follow the engine's rules. P then predicts from that state. It covers teams within 6 tiles of our units; farther teams cannot reach them by ply 2. Its drawn moves and aim replace the scripted steps and attention weights. Every state is checked against the search (units where it settled them, rivals it kills dead). Playouts are now 2 plies; the scripted plies 3–4 are gone. | P on the simulated next turn (v12's own step: HP, deaths at HP 0, pickups, drops, captures and now attack events): rival orders and forecasts. This replaces the scripted rival moves and the hand-written rival fire allocation. |
| Checked lookahead branches | P on each simulated successor (stepExact), with the root as its previous turn: rival orders, forecasts, follow-up plan | (the v12 rollout above) |
| Duel planner (2 teams, C08) | P marginal, P(attack) and aim at our units. The focus model and the last-target habit are gone. The 24-sample refinement, which never ran at 2 teams, is deleted. | (single planner) |
| Opponent drop rates | P(drop) per move. The learned per-team drop counts are gone. | P(drop) via the act head |

**History contract.** The network's history features (each unit's previous cell) always come from the previous **real** turn. Simulated states are predicted in separate views and never written into the real history. This fixes the C08 Fusion defect where branch states leaked into the next real turn. `contract_test.cpp` checks it on every turn of the recorded games.

**State contract.** Every state handed to the network keeps the rules' invariants: no dead unit on the board, an off-board unit carries nothing, and every carried flag has a living holder that carries it. Every C08 second-turn state also agrees with the search. Violations are counted (`inconsistent`, must stay 0). With the recorded orders, both bots' turn kernels (C08 stepExact, v12 simStep) reproduce the engine's next state field by field: positions, HP, carried flags, respawn times, flags, scores and attack events. One exception: a flag due to return at a random spot is kept off the board in simulated states until it reappears; those states are counted in the "flag-return states" column.

**Kept on purpose** (our evaluation, not behaviour prediction): game rules (legal moves, settlement, range and line of fire, respawns, flag returns), our task assignment and scoring, and C08's duel carrier-route risk, which checks which enemy cells can reach the route. The task geometry for intercepting an enemy carrier still assumes it heads for its base; TASK.md asks whether that should change too.

## Verified here (macOS laptop; no timing or strength claims)

- **Compiles:** both P-only carriers compile with `-Wall -Wextra` at sizes S to XXL. C08 shows only the 4 warnings its original already has; v12 shows none. The C08 control without PONLY_STATS compiles to assembly identical to the original.
- **Tests pass:** `python -m unittest discover -s bots/p_only/tests` runs 10 tests and all pass (~20 s).
- **Contract test:** zero failures and zero inconsistent states over about 49,000–147,000 checks per recorded game. The 4-team stream exercised 854 C08 and 2,666 v12 simulated-branch views. No prediction fell back to uniform.
- **Engine comparison:** both kernels reproduced the engine on all 706 comparable turns (351 in the duel, 355 in the 4-team game; 92 turns with a pending random flag return were skipped). v12's simulator left dead units at negative HP; that is fixed in the P-only build.
- **Smoke screen:** 10 games (2 and 5 teams, 1 s limit) through the full pipeline, 0 faults, 0 inconsistent states.
  - Network calls per turn of our seat, averaged over those 2 games: v12 P-only M about 8 views and 21 team encodes; C08 P-only M about 243 views, 622 team encodes and 607 action heads.
  - Every C08 arm reached the full 512 samples per expectimax call at these counts, so the time budget did not bind there.
  
  These are 2 games: they show the code runs, not that it is stronger.

## Known limits (read before interpreting results)

- **No legacy blend:** the fitted mixtures gave P 96–99% weight. P-only drops the remaining legacy weight entirely and adds no uniform floor. Calibration of pure P on tune data is open.
- **C08's search changed shape:** its playouts are now 2 plies, and the same wall-clock budget may buy fewer samples. `c08_control_2ply` and the samples-per-call column separate depth and search time from the change of predictor.
- **One second-turn prediction per sample:** C08 takes it from the state the best plan reaches and shares it across all candidate plans. That is an approximation; TASK.md Phase 4 measures it before any four-turn work.
- **V is experimental:** the `_m_v` variants are experimental (`value_approved` is false in the manifests). Screen P first.
- **Timing comes from the host only:** the timings above come from a laptop and are not evidence. Use the host screen's fault, late-reply and max-reply columns.
