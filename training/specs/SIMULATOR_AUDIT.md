# Simulator and telemetry audit (Task A)

Audited code is AI_Test `main` (85e9b69): `ctf-arena/Player7/workspace/experiments/round3/nextgen/` (`master.cpp`, `search/*.hpp`) and the official engine `ctf-arena/Player7/workspace/engine.py`.

**Findings in brief:**
- `stepExact` is rule-exact, but it is welded to planner state and gives up on every turn where a flag returns. In real play that is about 10% of turns, and most of those flags are grabbed at once.
- The search rollout is exact only for ply-1 movement. Fire is an expected-value model; there is no rival-on-rival damage and no rival flag transitions; plies 2+ are scripted with no settlement.
- `inference/sim.hpp` is the decoupled engine-exact simulator. It is verified on every turn of 29 recorded matches, plus the 18 matches of every test-suite run.
- Telemetry keeps requested moves separate from resolved positions.
- The workspace `core.py` cannot run matches on Windows; `fusion/runner.py` replaces it.

## A1. `stepExact` (master.cpp, `multi::Planner::stepExact`)

| Aspect | Finding |
|---|---|
| Rules | Matches engine phases 1–6 for full joint orders: movement settlement, simultaneous attacks with pre-pickup flag state, deaths and flag drops, pickups in id order, voluntary drops, captures. AI_Test's 179 kernel cases check it against `engine.py`. |
| Coupling | A planner member. It calls `respawns(g)`, which writes the planner's `pos` and `immune`, and reads `nav`, `shot()`, `immune` and the `Game` copy. Callers must wrap it in `saveRuntime()` / `restoreRuntime()` (the checked lookahead does this). |
| Flag returns | Returns `false` when any flag returns this turn, because the engine draws the spot from a hidden seed. **In 2 recorded real-bot matches (800 turns), 9.9% of turns had a return, and 68% of returns were picked up in the same settlement by a unit camping the spot.** The lookahead is blind on exactly the turns where a flag changes hands. |
| Cost | Copies the whole `Game` (string statuses, vectors) per step and records only attack and capture events. Fine for a few lookahead steps; too heavy for sampling. |

## A2. Search rollout simplifications (`search/rollout.hpp`, `search/dynamics.hpp`)

### Ply 1, against the engine

| Engine | Rollout |
|---|---|
| Movement settlement | **Exact** (`dynamics::settle`: shared destinations, cross-team swaps, chain blocking to a fixed point) |
| Each attacker shoots one requested target | **Expected value:** each armed rival fires with probability `fire[pick]` (1 if its move was blocked) and splits attention over every unit it can hit by `attentionWeight(hp, carrying, distance)`. Hit counts are convolved and death is P(hits ≥ lethal). |
| Rivals shoot each other too | **Ignored:** other teams' units only dilute attention; rivals never take third-party damage. |
| Our volley | **Exact** for the plan's targets (`dynamics::volley`) |
| Rival pickups, drops, captures; flag status changes | **Not modeled.** Our own pickup and delivery enter as survival-gated values; flag state is not updated. |
| Flag returns (random spot) | Not modeled |
| Damage 34, HP 100 from INIT | Hard-coded `34` (`34 * hits`, `(hp + 33) / 34`); equal to the official rules |

### Plies 2..H (H = 2, or 4 with `MOD_PLAYOUT`)

| Engine | Rollout |
|---|---|
| Rivals choose moves | **Scripted:** carriers step greedily toward base; others repeat their last displacement, or stay if blocked |
| Settlement and collisions | **None:** units may overlap; no blocking |
| Rival deaths after ply 1 | Not removed; rival HP not reduced after ply 1 |
| Our units | Independent greedy step per unit, no joint coordination; expected HP and survival probability carried forward |
| Pickups, captures, drops, respawns | Not simulated; the evaluator's potential stands in for them |

The design brief's claim — "rollout simplifies opponent casualties and flag transitions beyond step one" — is accurate. Even ply 1 omits rival-on-rival fire and rival flag transitions.

## A3. Decoupled simulator: `inference/sim.hpp`

`fusion::step(board, rules, state, orders, next, workspace, events, returnSpot)` is `stepExact` without the planner:
- **Inputs:** a plain `fusion::State` and one `Order` per unit (move, act, target).
- **Output:** the next state and its attack events.
- **Allocation:** buffers live in `Workspace`; after first use no step touches the heap (`tools/cpp/runtime_check.cpp`).
- **Flag returns:** the caller may pass `returnSpot[f]`. Without it a returning flag comes back `home` at (-1, -1), and `unknownReturns` counts it.

The engine picks the spot uniformly among flag spots (`SPOTS`) not occupied by a home flag, one returning flag after another. A caller can therefore sample from exactly the engine's distribution instead of giving up.

**Verification.** `tools/cpp/sim_check.cpp` replays recorded matches: recorded orders → `step` → compare with the next recorded block (units, flags, scores, attack events). It compares:
- 27 test-bot matches (2–4 teams);
- 2 real-bot matches (Sol, Fable, Astra, Opus; 3 and 4 teams; 509 deaths, pickups, drops, captures and respawns);
- the 18 suite matches in `tests/test_pipeline.py`.

Result: **every turn identical, 0 mismatches, 0 skipped.** Returning spots come from the recorded block, from the picker's cell when a camper took the flag at once.

**Coverage is part of the result.** A turn whose returning spot the next block does not reveal is skipped, and a replay with no compared turn proves nothing. `sim_check` therefore prints compared and skipped turns, exits 3 when nothing was compared, and the pipeline test requires every turn of every match to be compared. `tools/host/env_check.py` reports both counts per match.

**Uses:**
1. **Checked lookahead.** Replace the member function plus save/restore, and stop refusing return turns (sample the spot).
2. **Search.** Exact multi-ply transitions for sampled joint orders (rivals from q_θ joint samples), instead of scripted overlap. This is the precondition the brief sets for deeper plies.
3. **Value data.** Off-policy states: branch any recorded state with alternative ego orders and recorded or predicted opponent orders, for one or more settlements, without running bot processes.

## A4. Requested moves versus resolved positions

**Collection** (`training/collect.py`) stores each reply line exactly as sent (`replies[t][seat]`), separately from the TURN blocks that hold resolved positions.

**Labels** (`fusion/labels.py`) come only from reply lines; see [ACTIONS.md](ACTIONS.md) for canonicalization. A move stopped by a collision keeps its requested direction, and `blocked` marks it: 13 of 2,800 rows in the real-bot matches. The tests check that every blocked unit really ended where it started.

**Missing replies.** Timeouts and malformed lines are `null` / no label, never "stay".

## A5. Running bots headlessly on Windows

| Item | Finding |
|---|---|
| `ctf-arena/Player7/workspace/core.py` (English) | Reads bot stdout with `selectors` on pipes. On Windows `select` accepts only sockets, so it raises on the first turn: **matches cannot run there.** |
| `ctf-stress-pack/arena/core.py`, `run_stress.py` (used by the 44k / 90k benchmarks on Windows) | Reader threads plus queues: Windows-safe |
| The two `engine.py` copies | Rule-identical: an AST comparison differs only in translated error messages |
| `fusion/runner.py` (this repository) | Mirrors the original thread-based core: discards stale lines, strikes, three-strike exit, `-std=c++20 -O2 -static` on Windows, w64devkit discovery, `.exe` suffix. `collect.py` and `env_check.py` use it. |
| Text vs bytes | Bot pipes are text mode like the official runner, so bots see the same input as in the arena. Every internal tool pipe (extractor, forecast replay, parity, simulator) uses bytes, so no CRLF translation on Windows. |
| Files | UTF-8 everywhere; POSIX-style paths in manifests; `os.replace` for atomic writes; spawn-safe multiprocessing |

**Not verified on Windows:** this machine is macOS. `python tools/host/env_check.py --ai-test <AI_Test>` on the Windows host covers it: build every pool bot, run headless matches, check the protocol, measure latency and the simulator. See [ENVIRONMENT.md](ENVIRONMENT.md).
