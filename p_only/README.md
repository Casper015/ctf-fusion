# P-only bots: the new version with no hand-written opponent prediction

This folder holds everything needed to build and test the new bots. They are the two carriers, C08 Player5_Omega_LIT and Player5 v12 (rollout_volley), with **every prediction of what opponents will do coming from the trained policy network P**. The hand-written forecasts, the habit and target guesses, and the scripted opponent moves in the lookahead and playouts are deleted, not just switched off. Our own planning is unchanged: task assignment, candidate moves, scoring, search and evaluation.

The host agent's instructions are in [TASK.md](TASK.md). Start there after the quick start below.

## Quick start (host)

The commands run from the repository root, with the AI_Test checkout in `AI_TEST_ROOT` (or pass `--ai-test`), Python 3.10+, NumPy and a C++20 compiler (w64devkit on Windows). They need no edits.

```sh
python -m unittest discover -s p_only/tests -v          # ~15 s: no hand-written identifiers, bundles, contract on recorded games
python p_only/tools/build.py --stats                     # ~1 min: 12 bundled single-file bots + build/p_only/manifest.json
python p_only/tools/run_screen.py --smoke                # ~1 min: 8 games, relaxed limits, checks the chain (not evidence)
python p_only/tools/run_screen.py                        # the paired screen (p_only/profiles/screen.json), official limits
```

The screen writes `p_only/report/screen/SCREEN.md`, `screen.json`, `RUN.json`, the schedule and the build manifest. Commit those; raw results and match logs stay in `data/p_only/screen/` (not committed). If the run is interrupted, re-run the same command: finished games are kept. The analysis exits with status 2 and marks the report INCOMPLETE while any arm misses a fixture. Do not delete `build/p_only` during a screen: result rows are matched to the binary hashes in its manifest.

The offline structure scan is already generated at [report/STRUCTURE_SCAN.md](report/STRUCTURE_SCAN.md). Re-run it with `python p_only/tools/structure_scan.py`; it plays no games.

## What is in the folder

| Path | What it is |
|---|---|
| `include/ponly.hpp` | The P-only opponent model. A `View` holds the network's predictions for one state (the real root or a simulated state): joint team moves and unit marginals, P(act \| move), P(target \| attack, move), mode and seeded draws. With no model it falls back to uniform over what the rules allow, and counts every fallback. |
| `bots/c08_ponly.hpp`, `bots/v12_ponly.hpp` | The new carriers. Each header comment lists exactly what was removed and kept. |
| `bots/variants/*.cpp` | Thin wrappers: policy size S, M or L, and the experimental value term (`_m_v`). |
| `profiles/variants.json` | All buildable bots: the P-only variants plus the unchanged controls and the earlier mixture ("hybrid") integrations from `candidate/`. |
| `profiles/screen.json` | The paired screen: arms, re-run (A/A) copies, comparisons, seed 20261031, 20 maps per count. |
| `tools/bundle.py` | Inlines a bot's local includes into one `.cpp`, the file a judge compiles and the identity a screen records. |
| `tools/build.py` | Bundles and compiles the variants and writes the manifest (bundle and binary SHA-256, every inlined file's SHA-256, compiler, flags). |
| `tools/run_screen.py` | build, `roster.strength` schedule and run (identical fixtures for every arm), then analyze. |
| `tools/analyze.py` | Paired differences per team-count band with 95% intervals, re-run noise, faults, late replies and max reply time, and network calls per turn. |
| `tools/structure_scan.py` | Offline triage of all 258 eligible bots: opponent model, lookahead, sampling, time budget, habits, joined with the light screen. |
| `tools/record_stream.py` | Records the input a seat receives in a game (the test fixtures). |
| `tools/derive/make_*.py` | How the two P-only headers were derived from `candidate/*_Fusion.hpp`, with every splice checked. A test re-derives them and compares. |
| `tests/` | `test_p_only.py` (Python), `contract_test.cpp` (C++ contract), `fixtures/stream_n02.txt.gz` and `stream_n04.txt.gz` (two recorded games). |
| `report/` | Generated results: the structure scan now, the screen later. |

## What the network now predicts, and where

| Place | C08 P-only | v12 P-only |
|---|---|---|
| Root forecast (every opponent unit, every legal move) | P marginal; P(attack), P(pickup), P(drop) after the move; P(attack and target = each of our units) | P marginal; P(attack); aim at each of our units |
| Incoming fire in our joint-move scoring | P's aim at our unit, counted only when the settled cells allow the shot. This replaces the priority, crossfire and attention weights and the rule that an enemy always shoots. | Same |
| Threat map (carrier routes, escorts) | P(move) × P(attack) over the cells each destination can hit | Same |
| Expectimax samples (ply 1) | One joint draw per team from P per sample, the same draw for every plan | (no expectimax) |
| Ply 2 of every rollout | Once per sample, P runs on the ply-1 state the sample reaches under the best plan so far. It covers teams within 6 tiles of our units; farther teams cannot reach them by ply 2. Its drawn moves and aim replace the scripted steps and attention weights. Playouts are now 2 plies; the scripted plies 3–4 are gone. | P on the simulated next turn: rival orders and forecasts. This replaces the scripted rival moves and the hand-written rival fire allocation. |
| Checked lookahead branches | P on each simulated successor (stepExact), with the root as its previous turn: rival orders, forecasts, follow-up plan | (the v12 rollout above) |
| Duel planner (2 teams, C08) | P marginal, P(attack) and aim at our units. The focus model and the last-target habit are gone. The 24-sample refinement, which never ran at 2 teams, is deleted. | (single planner) |
| Opponent drop rates | P(drop) per move. The learned per-team drop counts are gone. | P(drop) via the act head |

**History contract.** The network's history features (each unit's previous cell) always come from the previous **real** turn. Simulated states are predicted in separate views and never written into the real history. This fixes the C08 Fusion defect where branch states leaked into the next real turn. `contract_test.cpp` checks it on every turn of the recorded games.

**Kept on purpose** (our evaluation, not behaviour prediction): game rules (legal moves, settlement, range and line of fire, respawns, flag returns), our task assignment and scoring, and C08's duel carrier-route risk, which checks which enemy cells can reach the route. The task geometry for intercepting an enemy carrier still assumes it heads for its base; TASK.md asks whether that should change too.

## Verified here (macOS laptop; no timing or strength claims)

- **Compiles:** both P-only carriers compile with `-Wall -Wextra` at sizes S, M and L. C08 shows only the 4 warnings its original already has; v12 shows none. All 12 bundled variants compile as single files (about 1 minute).
- **Tests pass:** `python -m unittest discover -s p_only/tests` runs 8 tests and all pass (~15 s).
- **Contract test:** zero failures over about 49,000–140,000 checks per recorded game. The 4-team stream exercised 864 C08 and 2,406 v12 simulated-branch views, so the history check is real. No prediction fell back to uniform.
- **Smoke screen:** 8 games (2 and 5 teams, 1 s limit) through the full pipeline with 0 faults. Network calls per turn of our seat, averaged over those 2 games:
  - v12 P-only M: about 8 views, 20 team encodes and 114 action heads;
  - C08 P-only M: about 243 views (mostly the ply-2 views of the sampled search), 609 team encodes and 595 action heads.
  
  These are 2 games: they show the code runs, not that it is stronger.

## Known limits (read before interpreting results)

- **No legacy blend:** the fitted mixtures gave P 96–99% weight. P-only drops the remaining legacy weight entirely and adds no uniform floor. Calibration of pure P on tune data is open.
- **C08's search changed shape:** its playouts are now 2 plies, its ply-2 predictions use one reference plan per sample, and the same wall-clock budget buys fewer samples. "P-only versus control" therefore also changes search depth and sample count. TASK.md asks for the diagnostics that separate these.
- **Simulated states are incomplete:** they carry no previous-turn attack events. C08's ply-2 sample views also keep the root's HP.
- **V is experimental:** the `_m_v` variants are experimental (`value_approved` is false in the manifests). Screen P first.
- **Timing comes from the host only:** the timings above come from a laptop and are not evidence. Use the host screen's fault, late-reply and max-reply columns.
