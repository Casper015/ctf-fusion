# Task for the host agent: P-only bots, then near-large / far-small P

Written 2026-10-09 against main at `99ecbe1`, for the agent that builds and runs bots on DESKTOP-G60OJUD.

You have two jobs, in this order:

- **Job A:** build bots whose opponent prediction comes only from the new P. Delete the hand-written prediction and the habit code, don't just switch them off.
- **Job B:** in multi-team games, work out whether a larger P for near opponents and a smaller P for far ones has any positive effect. Then test it if the offline evidence says it might.

Re-read the repository first (Step 0). Several things you need are already there, and some earlier assumptions are wrong.

## Rules that still apply

- **Runtime:** bots are CPU-only C++. The first reply is limited to 2 s and every later reply to 50 ms. That deadline is the only hard limit; source size and memory are free.
- **English only:** every repository file is in English. The `卸掉V` comments that `0a0badc` added to the variant `.cpp` files and the V ablation report should be replaced.
- **Git:**
  - Commit directly to `main`, one commit per stage, and don't commit unrelated files.
  - Never use a bare `git stash`.
- **Report what the data shows:**
  - A point estimate whose interval includes zero is "not shown", not "an increase".
  - Use "rank #1", "significant" or "conclusive" only when the interval supports it.
  - Missing timing stays missing, never zero.

## Step 0: re-read the repository

Read these in order:

1. `reports_m/review_bundle_20261009/README.md` and `CURRENT_STATUS_AND_V_ABLATION_REVIEW.md`: the open defects, which are C08 history, adapters and records.
2. `handoffs/diverse15_distance_tiered_policy/DISTANCE_TIERED_POLICY_PLAN.md`: the existing plan for exactly the Job B idea. It covers the spawn-distance prior, dynamic overrides, budget admission and exact caches. Build on it rather than starting over.
3. `candidate/C08_Player5_Omega_LIT_Fusion.hpp`, `candidate/Player5_v12_Fusion.hpp` and `tools/build_carrier_variants.py`: how P is wired into the bots today.
4. `inference/adapter.hpp`, `inference/policy.hpp` and `inference/state.hpp`: the P API (`jointPolicy`, `actPolicy`, `targetPolicy`, `drawTeam`, `drawAct`) and the history contract in `makeView`.
5. `reports/capacity_DESKTOP-G60OJUD.md`: the measured P and V cost per tier on this host.
6. `reports/tournament_1000_bt_report.md`, together with the notes on it below.

Then write `reports_p_only/EARLIER_TESTS.md`. Give one row per earlier experiment with:

- its commit, seed and team counts;
- the number of maps and games, and which bots played;
- what it can support and what it cannot.

Keep the earlier tests and the new work apart:

- New bots go in `candidate/p_only/` and new results in `reports_p_only/`.
- Don't edit old reports beyond adding a status line.

These facts were checked on `99ecbe1`. Verify them yourself; don't trust them blindly.

**No P-only bot exists yet.**
- Both carriers blend P with the hand-written forecast. The fitted alpha is 0.96–0.99 for M and L, so at the root P already carries most of the movement forecast, and `alpha = 1` alone changes little.
- The hand-written prediction still drives most of the rest:
  - the shoot and drop chances;
  - C08's scripted opponent continuations: `rivalOrders` models 0/1 (used by `refineChecked` and by the V branch), and `scriptedStep` in playout plies 2–4;
  - v12's `rivalWants` models 0/1;
  - C08's duel predictor, `duel::Player::predict`;
  - the legacy component of the joint sampler.

**Habit code:**
- C08's habit modules (`habits.inc`, `habits_duel.inc`, `habits_multi.inc`) are compiled out (`MOD_HABITS 0`), but their code is still in the header.
- v12 has no habit module.

**`candidate/bot_p_s.cpp` contains no P.** It is byte-identical to `candidate/bot_baseline.cpp`. Delete it or rename it, and never use it as a P bot.

**C08 corrupts its own history.**
- The chain is `refineChecked` → `planTurn(next,false)` → `fSession.update(next)`, which writes simulated states into the persistent session. `saveRuntime` and `restoreRuntime` don't cover that session.
- The next real turn can then read a simulated state as its previous turn.
- P-only puts P inside the branches, so fix this first.

**Re-run noise is large for C08.**
- Re-running the unchanged C08 control on the same 80 maps reproduced the score in only 36 of 80 games, and only 2 of 20 at 15 teams. v12 reproduced 68 of 80.
- The cause is C08's timer-bounded search.
- C08 comparisons therefore need repeats of the identical bot (A/A runs) and more maps.

**The `99ecbe1` tournament doesn't separate the bots.** These BT differences come from a bootstrap over maps within each team count, the same method as `bt.py`:

| Comparison | BT difference | 95% interval |
|---|---:|---:|
| C08 pvL − control | +28.9 | −12.7 to +69.8 |
| v12 pvL − control | +21.7 | −18.4 to +59.0 |
| C08 pvL − v12 pvL | +23 | −16.4 to +64.8 |

- None of these intervals excludes zero.
- 100 of its games reuse maps from the seed-20261020 pilots.
- Its 0 faults over 1,000 games is real evidence: in `fusion/runner.py` a late reply counts as a fault.

**Seeds already used:** 20261009 (light screen) and 20261020 (P pilot, V ablation and tournament). Use new seeds, and never tune and confirm on the same maps.

## Job A: P-only bots

**Definition:** every prediction of what opponents will do comes from P. The bot's own planning stays unchanged: task assignment, candidate moves, scoring and the settlement simulator. Only the opponent model changes.

Delete the following from the P-only headers; it is not enough to switch them off.

| Path | Today | P-only replacement |
|---|---|---|
| C08 multi: `Planner::forecast` | Legacy outcome probabilities mixed with P marginals | P marginals |
| Shoot and drop chances, in both carriers | Heuristic. C08 blends its shoot chance with P's attack probability; v12 uses no P here | P conditional action head (`actPolicy`) at each predicted move; targets from `targetPolicy` |
| C08 `rivalOrders` models 0/1 (`refineChecked`, V branch) | Mode of, or sample from, the legacy forecast | Model 0 = P joint mode; model 1 = P joint sample (`drawTeam` + `drawAct`) |
| C08 `search::JointSampler` | Joint-v2 mixture including the legacy component | P joint draws only |
| C08 `scriptedStep` (playout plies 2..4) | Scripted step: carriers go home, others repeat their last move | Stage 1: keep it, labelled as scripted. Stage 2: P on the settled successor state |
| C08 `duel::Player::predict` | Hand predictor mixed with P marginals | P marginals. The duel refinement block never runs at N=2 because Expectimax is off for duels; decide what it should do and document the choice |
| v12 `forecast` | Legacy forecast mixed with P marginals | P marginals |
| v12 `rivalWants` models 0/1 | Legacy mode on ply 1, then hand-scored moves | P mode or P sample on the settled state |
| C08 `habits*.inc` code | Compiled out | Deleted |

Build two stages, each as a separate bot:

- **P-root:** every prediction made at the root comes from P. Later playout plies may stay scripted, and the bot's name must say so.
- **P-full:** continuation plies and branch states also use P. This requires branch-local sessions: each branch state is encoded with its own parent as history, and the root session never changes during search.

**Replace the legacy weight with a uniform floor.** Setting `alpha = 1` removes the 1–4% legacy weight that currently softens P's overconfident mistakes. Replace it with a small floor that spreads uniformly over legal moves, with its size fitted on tune data. A uniform floor is not hand-written prediction. Report the integer NLL per team count for three versions: P alone, P plus the floor, and the old mixture.

**No-model fallback:** if a team count has no model, play uniformly over legal moves, and count every time this happens. A P-only build must not compile the legacy forecast at all. Add a test that fails if it does, for example by searching the preprocessed P-only translation unit for the legacy function names.

**Arms:** V stays off in all of them; V becomes its own arm later.

- control, which is the current hybrid;
- P-root at S, M and L;
- P-full at S, M and L;
- XL and XXL, added only after the timing check.

The builder writes all arms under `candidate/p_only/` with a manifest. The manifest records hashes of the source, the included headers, the weights and the binary.

**Checks before any matches.** Compiling and unit tests can run anywhere; matches and timing run on this host only.

- `-O2 -Wall` builds cleanly, and the existing tests still pass.
- **Root-session regression:** enumerating branches in a different order or number must leave three things identical to a clean replay:
  - the root features;
  - the root P output;
  - the next real turn's features.
- **Branch encoding:** encoding a branch state with its parent gives the same features as a fresh encode of that state.
- **Timing at 15 teams with 16 seats running at once:**
  - complete-turn p50, p99 and max;
  - first reply;
  - late count;
  - P calls per turn.

## Job B: should near opponents get a larger P and far ones a smaller P?

Think it through before building anything. These starting points come from the committed measurements; verify each one.

1. **V is not per opponent.** V scores the whole state once per search node, so "large V for near opponents, small V for far ones" doesn't exist. The closest V question is a larger V at the root and first ply with a smaller one deeper in the search. Treat that as a separate experiment.
2. **At the root, tiering saves almost nothing.**
   - Root P for all 14 opponents at 15 teams costs: S 0.27, M 0.34, L 0.44, XL 0.70, XXL 1.34 ms.
   - Those figures are p99 with 16 seats running at once, computing every head.
   - Everyone-at-XXL already fits, so "near XXL, far S" saves under 1 ms per turn.
   - Feature encoding is a large share of the small-tier cost, and tiering doesn't reduce it.
   - The baseline to beat is "every opponent at the largest tier that fits", not "everyone at S".
3. **Where tiering can matter is P inside search (P-full).**
   - Calls multiply. For example, 24 scenarios × 3 later plies × 14 teams is about 1,000 team-calls per turn.
   - At the XXL upper bound (about 0.1 ms per team) that is about 100 ms. At S (about 0.02 ms per team) it is about 20 ms.
   - Here tiering, together with exact caching, decides whether P-full fits in 50 ms at all, and how many scenarios it gets.
4. **Does a bigger P help more for near opponents?** Measure this offline first, with no matches.
   - On held-out records, compute each opponent team's NLL gain (L − S, XXL − S).
   - Group it by wall-aware distance from their units to our units, flags and carrier, by whether contact is possible within 2 turns, and by flag carrier.
   - If the gains concentrate near us, tiering is promising. If they are uniform, near-large gives nothing beyond all-large.
5. **Does the far prediction change our decisions?**
   - On recorded states, swap far opponents' predictions between S and L, and count how often the carrier's chosen orders change (the decision-flip rate). Do the same for near opponents.
   - If far flips are near zero, a small P for far opponents costs nothing.
6. **"Near" must be dynamic.**
   - Use the current wall-aware distance to our units, flags and carriers, contact within the search horizon, and overrides for flag carriers and threats.
   - Spawn distance is only the starting prior, as in the existing plan.
   - Add hysteresis so a team's tier doesn't flicker, and break ties by team id.
7. **Fit each tier to the opponents it actually serves.** Each tier was trained on all opponents. If a tier only ever sees far (or only near) opponents, fit its floor or temperature on tune data from that bucket, and gate it on that bucket.

Run the play test on the host only if points 4 and 5 suggest an effect:

- **Arms**, all on the P-full carrier at the same per-turn CPU budget:
  - all-S;
  - all-L;
  - every opponent at the largest tier that fits;
  - near-L / far-S;
  - near-XL / far-S.
- **Reinvest the saved time:** the tiered arms must spend it on more search scenarios. Otherwise they can only tie or lose.
- **Team counts:** 3–15 only, since a duel has only one opponent. Report by band: 3–5, 6–10 and 11–15.
- **Records:**
  - which tier each opponent got on each turn;
  - the flip rates;
  - P calls per turn;
  - full timing.

My expectation, to be checked rather than assumed, is no measurable effect at the root. A gain is possible only in P-full, where tiering buys more scenarios inside 50 ms.

## How to run the matches (host)

- **Paired fixtures:**
  - The same map, flag seed, opponents and seat; only our bot changes.
  - Don't use random seating. The 1,000-game random-seat tournament could not tell +29 BT from zero.
- **New seeds:**
  - Use 20261031 for tuning.
  - Choose and freeze a confirmation seed later, and don't look at its maps until the finalists are frozen.
- **Opponents:** use a fixed field, for example the light-screen field, and record the list.
- **Screening size:**
  - Use about 300 paired fixtures per comparison, spread over the team counts. The observed per-fixture SD of about 0.25 gives a standard error of about 0.015 utility.
  - Also run A/A repeats of the control on 100 fixtures to measure re-run noise.
- **Confirmation:** a finalist is confirmed under the review bundle's rule of at least 10,000 independent map groups over team counts 2–15.
- **What to save:**
  - the runner's `first_ms`, `p99_ms`, `max_ms`, `late` and `faults`, with fault reasons;
  - hashes of the source, headers, weights, binary, engine and schedule.
- **Missing games:** if a game is missing, the run fails rather than silently dropping it.

## Report back

1. The `reports_p_only/EARLIER_TESTS.md` table.
2. For the P-only build: what was removed from each carrier, the test results and the timing.
3. Offline answers to Job B points 4 and 5, with numbers.
4. Paired results with intervals per band, the A/A noise, and a plain verdict for each comparison: helps, hurts or not shown.

Commit each stage to `main` separately, with a message that says what the stage does.
