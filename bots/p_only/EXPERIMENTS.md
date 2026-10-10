# Controlled P/V experiments at two plies

This is the experiment matrix for the next host runs: does a larger P, more search samples, or the value network V make the complete bot stronger under the 50 ms deadline? Each comparison changes one factor, or measures an interaction on purpose. Depth stays at two plies; four plies and the final confirmation are later experiments.

Nothing here is a strength result. The verification at the end lists only what was observed on a development machine (compiles, tests, replays, smoke games), not timing or strength.

## What changed in the bots

- **Fixed-sample search (C08).** `PONLY_FIXED_SAMPLES=N` makes every expectimax call ask for exactly N joint samples instead of filling the 30 ms search budget.
  - The hard turn deadline still binds: the 35 ms compute target that the deployed bot also uses.
  - A call that reaches it stops and is counted as truncated. In this mode the checked lookahead keeps only that hard deadline, and a lookahead it cuts short is counted too.
  - The default, `PONLY_FIXED_SAMPLES=0`, is the deployed fixed-time mode, unchanged.
- **Search counters (C08, `PONLY_STATS`).** One stderr line per game: expectimax calls, samples completed, samples requested, truncated calls, plan evaluations (rollouts), plans, plans still compared at the end (survivors), checked lookaheads, and lookaheads cut short. Both C08 controls print calls, samples, requested, truncated and plans.
- **Independent P and V sizes.** `training/export_mixed.py` builds a model set from one tier's P and another tier's V.
  - It copies the committed integer arrays and checks every array byte for byte; nothing is retrained.
  - `inference/models_pxl_vm/` is P from XL with V from M; `inference/models_pxxl_vm/` is P from XXL with V from M.
  - Each set is the only registry in its binary, so the tiers' shared include guards and namespaces never clash.
  - Each V-off/V-on pair builds from the same set; the wrappers differ only in `MOD_FUSION_V`.
- **The V consumer.**
  - **Settled states:** V is evaluated on fully settled successor states (C08 stepExact, v12's simulator).
  - **Several scenarios:** where there are several, each scenario's realized state gets its own V. The values then combine the way the heuristic combines scenarios: mean and worst in C08's checked lookahead, 0.7/0.3 in v12's rollout.
  - **Game over:** once the game is over, the exact pairwise utility from the scores replaces V.
  - **Beam:** C08's one-ply beam re-ranking uses a single scenario, every rival on its most probable order. It now re-ranks only the plans that survived the search's final round. Sequential halving values eliminated plans on fewer samples, and with V at weight 0 the old hook changed 5 of 400 replies on a recorded game.
  - **Coefficients:** they are build settings, treated as tuning parameters: `PONLY_V_REFINE` (50), `PONLY_V_BEAM` (35), `PONLY_V_ROLLOUT` (40).
  - **V-off baseline:** the V-off build is the unchanged heuristic evaluator.
  - **Counters:** per path, V calls, exact terminal utilities, decisions, decisions V changed against the same path without V, and seconds spent in V.
- **Interleaved, identity-bound runner.** `bots/p_only/tools/screen_runner.py` plays fixture by fixture, every arm on the same map, flag seed, opponents and seat. The arm order rotates by fixture, so every arm takes every position equally often.
  - Each row carries an identity digest covering:
    - bundle and binary SHA-256;
    - every inlined source with its SHA-256 (carrier, P-only model, inference headers and the model weights);
    - the runner code;
    - the schedule and the limits.
  - Resuming skips only rows of the identical identity; the analysis treats any other row as stale and reports missing fixtures.
  - An A/A arm (`X~aa`) replays X's identical binary and identity.

## Stage 1: capacity and sampling (`bots/p_only/profiles/screen_capacity.json`)

Seed 20261107 (fresh), 20 maps for each count 2–15, 280 fixtures. V is off in every arm.

| Arm | What it is |
|---|---|
| `c08_control_2ply` | Unchanged C08, 2-ply scripted playouts (the depth-matched control) |
| `c08_ponly_xl`, `c08_ponly_xxl` | P-only, fixed time (the deployable configuration) |
| `c08_ponly_xl_fs256`, `c08_ponly_xxl_fs256` | P-only, exactly 256 samples per call (equal-work diagnostic) |
| `v12_ponly_xl`, `v12_ponly_xxl` | v12 references |
| `c08_ponly_xxl~aa`, `v12_ponly_xxl~aa` | Re-runs of the earlier screen's P-only finalists (identical binaries) |

9 arms, 2,520 games. The comparisons, in order:

| Comparison | What it measures |
|---|---|
| `c08_ponly_xl − c08_control_2ply`, `c08_ponly_xxl − c08_control_2ply` | The network replacing the script, at equal depth, as deployed |
| `c08_ponly_xxl − c08_ponly_xl` (fixed time) | The capacity trade-off as deployed: better predictions against fewer samples |
| `c08_ponly_xxl_fs256 − c08_ponly_xl_fs256` | Prediction quality at equal search work (valid for its untruncated calls; read the truncated share) |
| `c08_ponly_xl − c08_ponly_xl_fs256`, `c08_ponly_xxl − c08_ponly_xxl_fs256` | What filling the time budget adds over 256 samples |
| `v12_ponly_xxl − v12_ponly_xl` | v12 capacity |
| C08 versus v12 at XL and at XXL | Carrier comparison |
| `X~aa − X` | Re-run noise of the finalists |

Fixed-sample games are diagnostics, not deployment results: the opponents and the deadline are real, but our bot spends a different amount of time per turn.

## Stage 2: V at fixed P (`bots/p_only/profiles/screen_value.json`)

Seed 20261108 (fresh), 20 maps for each count 2–15, 280 fixtures, fixed-time deployment games. Every arm uses a mixed set with the same V-M.

| | V off | V-M on |
|---|---|---|
| P-XL (`models_pxl_vm`) | A: `*_ponly_pxl_vm` | B: `*_ponly_pxl_vm_v` |
| P-XXL (`models_pxxl_vm`) | C: `*_ponly_pxxl_vm` | D: `*_ponly_pxxl_vm_v` |

Each table applies to both carriers: 8 arms and 2,240 games.

| Contrast | Measures |
|---|---|
| B − A and D − C | V's increment at P-XL and at P-XXL |
| C − A and D − B | P capacity with V off and with V on |
| D − C − B + A | The interaction: does V's effect depend on P's size? |

The V coefficients are the experimental defaults, not tuned values. A null or negative V result says nothing about better coefficients. If you tune them, do it on tuning maps only and freeze them before any confirmation. Read the value table: a path with zero V calls was not tested.

## Host commands

```sh
python -m unittest discover -s tests -t .
python -m unittest discover -s bots/p_only/tests -v
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_capacity.json --dry-run   # builds, identities, job order
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_capacity.json             # stage 1, 2,520 games
python bots/p_only/tools/run_screen.py --config bots/p_only/profiles/screen_value.json                # stage 2, 2,240 games
```

Keep the worker count (6) and the host's other load fixed for a whole stage, and record them. Rerunning a command resumes. Don't delete `build/p_only` between runs: a rebuilt binary is a new identity, and its earlier rows become stale.

Each stage writes `bots/p_only/report/capacity|value/` with:
- `SCREEN.md`;
- `screen.json`;
- `RUN.json`;
- `identities.json`;
- the schedule and the build manifest.

The reports contain:
- paired differences and contrasts, by band (2, 3–5, 6–10, 11–15), with 95% intervals;
- re-run noise;
- per arm: faults, dead games, late replies, first-reply maximum, per-game p99 (median and worst game), and the largest single reply;
- network calls per turn;
- C08 search work (samples, requested, truncated share, rollouts, survivors, lookahead cuts);
- V use per path.

**Rules for reading the results:**
- Faulting or late games stay in the utility.
- Report an intervals-include-zero result as "not shown".
- Treat a single band's interval with care: the A/A rows show how often noise alone crosses zero.
- Report a fixed-sample comparison only together with its truncated share.

## Known limitations

- **Shared second-turn prediction (C08).** One prediction per joint sample comes from the state the best plan so far reaches, and every plan of the sample shares it. Common random numbers are justified; sharing predictions across different states is a measured-cost approximation that still needs its own error measurement.
- **Flags due to return.** A flag due to return at a random spot stays off the board in simulated states until it reappears; these states are counted as "flag-return states". No chance outcome is sampled for it yet.
- **V coverage:**
  - C08's duel planner has no V path (zero calls at two teams).
  - The one-ply beam re-ranking runs only when the checked lookahead won't, and uses one scenario.
  - The checked lookahead runs only in flag battles.
- **Fixed-sample games still spend wall-clock time.** They use the hard deadline, so a heavy turn can truncate them; the truncated share says how often.
- **Mixed sets have no new gate report.**
  - Their P and V arrays are exact copies from the gated exports (`reports/models/xl|xxl|m/`).
  - P's alpha comes from the P tier; the P-only bots don't use it.
  - V stays unapproved (`value_approved: false`).
- **Binaries aren't byte-reproducible across rebuilds.** The identity digest, which includes the bundle and every input, is what binds results.

## Verified on the development machine (not timing or strength evidence)

- **Tests:**
  - Repository suite: passes, including `tests/test_mixed_export.py`, which re-derives both mixed sets and checks every array against its sources.
  - P-only suite: 15 tests pass.
- **Contract test with engine comparison, on three recorded games** (2, 4 and 15 teams). The 15-team game has 1,985 attacks, 641 deaths, 227 pickups, 70 captures, 70 flag returns, 622 respawns and one voluntary drop.
  - 0 failures and 0 inconsistent states for both carriers.
  - Both bots' turn kernels reproduce the engine's next state field by field on every comparable turn: 333 turns at 15 teams; turns with a pending random flag return are skipped.
  - Every simulated state built during a turn has that real turn as its immediate parent.
- **Fixed-work replay on the recorded games** (C08 with P-XL/V-M at 64 samples; v12 with P-XL/V-M):
  - A build with V compiled in at weight 0 replies exactly as the V-off build, on 400 of 400 turns in both games and for both carriers. Before the beam fix, C08 differed on 5 of 400.
  - Turning V on never changed the network's root predictions (identical digests on every turn).
  - With V on at the default coefficients, C08 changed 41 of 400 replies in the 4-team game and none in the duel (no V path); v12 changed 36 and 29.
- **Dry runs:** both stages built every arm (9 and 8) and wrote their identities. A/A copies share their base arm's identity.
- **Smoke screens with relaxed limits:** 10 games through the interleaved runner and 2 games with V arms, 0 faults. They exercised the search and value counters and the exact terminal utility.
