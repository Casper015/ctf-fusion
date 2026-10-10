# Task for the host agent: P-only bots

Written 2026-10-09 for the agent that runs ctf-fusion on DESKTOP-G60OJUD. Read [README.md](README.md) first; it has the commands, what was built and what is already verified. Work in the phases below, in order, and commit each phase's results to `main` before starting the next.

**Order and status.** The second-turn state fix and the 2-turn scripted control (c08_control_2ply) are already in the code; don't screen anything built before them. Phase 0 can start now and depends on nothing else. Phase 1 adds the XL and XXL timing and the size rule. Phase 2, the screen, runs only after Phase 1 has applied that rule. Phase 3 does its offline check before building anything. Phase 4 is the four-turn lookahead. Phase 5 is V and the final confirmation. The XL/XXL evidence backfill runs on the GPU in parallel.

## Rules

- **Runtime:** bots are CPU-only C++. The first reply is limited to 2 s and every later reply to 50 ms. That deadline is the only hard limit; source size and memory are free.
- **English only:** every repository file is in English.
- **Git:**
  - Commit directly to `main`, and don't commit unrelated files.
  - Never use a bare `git stash`.
  - Commit reports, not raw data: `data/` and `build/` stay out of git.
- **Report what the data shows:**
  - An interval that includes zero means "not shown", never "an increase".
  - Missing timing stays missing, never zero.
  - Don't write "rank #1", "significant" or "conclusive" unless the interval supports it.
- **Fresh maps:** never reuse seeds 20261009 or 20261020. The tuning seed is 20261031; choose the confirmation seed later and keep it unseen until the finalists are frozen.

## Phase 0: offline structure analysis (no games)

Question: of all the bots we have, which give the strongest complete bot once P and V are added? Answer it from code and existing evidence before any new games.

1. **Read the triage table.** Start from [report/STRUCTURE_SCAN.md](report/STRUCTURE_SCAN.md), which covers all 258 eligible bots; 64 have light-screen results. Its signals are regular expressions: use them to choose what to read, not as conclusions.
2. **Read the code of the 8–12 strongest candidates.** Include v12 and C08, the R3/C20 HABITS family, cand-mc1/mc2, Control_external_CL, R3_EXPECTIMAX_sampled and S1_C_Adaptive6p_Fast15. For each, write down:
   - where the opponent forecast is produced and where it is consumed: root scoring only, or inside a lookahead, sampler or rollout;
   - how many future plies it simulates, and with what opponent model;
   - whether it samples several opponent futures, which is where P's joint distribution helps;
   - whether a leaf evaluation exists where V could score states;
   - its time budget and screen reply times: the headroom left for network calls;
   - how much hand-written prediction and habit code P would replace;
   - per team-count band (2, 3–5, 6–10, 11–15): the screen utility.
3. **Use the existing evidence, within its limits.** That means the light screen (`roster/report/screen_light`), the P pilot (e11c45b), the V ablation (0a0badc), the 1,000-game tournament (99ecbe1), and the reviews in `reports_m/review_bundle_20261009/`.
   - Their maps overlap.
   - The tournament could not tell its pvL-versus-control gaps from zero; its 95% intervals were −12.7 to +69.8 BT for C08 and −18.4 to +59.0 for v12.
   - C08's own control reproduced its score in only 36 of 80 reruns, so its comparisons need A/A repeats.
4. **Write `p_only/report/STRUCTURE_ANALYSIS.md`.** It should contain:
   - a ranked shortlist of 2–4 carriers for P-only integration, with reasons;
   - for each, the expected place where P and V add value;
   - the integration cost;
   - the risks.

   Say plainly when two candidates cannot be separated without games. If a carrier outside C08 and v12 ranks high, describe how to build its P-only version the same way: `include/ponly.hpp`, a derivation script, wrappers and a variants entry. Ask the user before building it.
5. **Inventory the earlier tests.** Add a section listing every earlier test with its commit, seed, team counts, maps, bots, games, what it can support and what it cannot. Keep earlier results where they are; new results go under `p_only/report/`.

## Phase 1: verify, then measure timing and apply the size rule

```sh
python -m unittest discover -s p_only/tests -v
python p_only/tools/build.py --stats
python p_only/tools/run_screen.py --smoke
python p_only/tools/run_screen.py --config p_only/profiles/timing.json
python p_only/tools/size_rule.py --apply
```

All tests must pass on Windows, including the contract test's engine comparison. If a test or a smoke game fails, stop and report it with the output.

**The timing run** plays 105 games: team counts 8, 12 and 15, 5 maps each, at the screen's 6 workers so the CPU load matches the screen. It covers the C08 control and C08 P-only at M, L, XL and XXL, plus v12 P-only at XL and XXL.

**The size rule** was fixed before the run (`p_only/tools/size_rule.py`):
- **C08:** the largest P-only size, from XXL down to M, whose worst per-game p99 reply is at most 40 ms (10 ms of headroom), with no fault and no late reply, enters the screen.
- **v12:** XL and XXL enter by default; a size with a fault or a late reply drops out.
- **Why one size per carrier:** tiering at the root saves almost nothing. At 15 teams, all 14 opponents cost 0.44 ms at L and 1.34 ms at XXL per turn, against a 50 ms budget.

`--apply` writes the chosen arms and comparisons into `p_only/profiles/screen.json`. Commit `p_only/report/timing/` with SIZE_RULE.md.

If no C08 size qualifies, don't screen C08. Reduce its second-turn work instead: share the view between samples whose turn-1 states are identical, or predict fewer teams. Report the change and rerun the timing.

## Phase 2: the paired screen

```sh
python p_only/tools/run_screen.py
```

Run it only after Phase 1 has applied the size rule.

**Design:**
- 20 maps for each of the 14 team counts (2–15), seed 20261031;
- 9 arms: C08 control (4-turn scripted), C08 2-turn control, C08 hybrid M, C08 P-only at the admitted size, v12 control, v12 hybrid M, and v12 P-only at XL and XXL;
- two re-run (A/A) copies of the controls;
- 3,080 games in total.

Keep the worker count fixed and record it.

**Report from `SCREEN.md`**, with a verdict per comparison (helps, hurts or not shown):
- **Script versus network at equal depth:** C08 P-only versus the 2-turn control. This is the cleanest test of "P replaced the script".
- **What depth alone did:** the 2-turn control versus the frozen 4-turn control.
- **Net change:** C08 P-only versus the frozen control.
- **Pure versus mixed:** P-only versus the hybrid.
- **v12:** P-only XL and XXL versus control, and XXL versus XL.
- **C08 versus v12.**
- **Noise:** the A/A rows (re-run noise and identical-score rate). Judge every C08 difference against them.
- **C08 diagnostics:** the network-calls table gives expectimax calls and samples per call (both C08 controls report them too), plus inconsistent states (must be 0) and states with a pending flag return.

If C08 P-only loses to the 2-turn control while running fewer samples per call, the loss may be search time, not prediction quality. Say so; don't conclude P is worse.

## Phase 3: near-large / far-small inside C08's second-turn views

This is not about multi-team games in general. At the root, every opponent can already have the largest size that fits (see Phase 1). The idea matters only inside C08 P-only's sampled search: about 240 network views and 620 team encodes per turn, almost all for the second turn of each sample, limited to teams within 6 tiles of our units. v12 makes only about 20 per turn.

1. **Offline first:** do larger P models help more for near opponents?
   - On held-out records, compute each opponent team's NLL gain (L − S, and XXL − S). Group it by wall-aware distance to our units, flags and carriers, by whether contact is possible within 2 turns, and by flag carrier.
   - On recorded states, swap far opponents' predictions between S and L and count how often C08's chosen orders change. Do the same for near opponents.
2. **Only if the gains concentrate near us,** build the tiered arm:
   - the root view stays at the admitted size;
   - in the second-turn views, teams in contact (distance 3 or less, or a flag carrier) get the large size and the rest of the 6-tile radius gets S;
   - this needs two registries in one binary, compiled at the larger buffer width.

   Hold the per-turn wall-clock budget fixed, and record samples per call, the size used per prediction, and the timing.
3. **Screen it on fresh tuning maps** against the single-size arm, for team counts 3–15, reported by band.

If the gains don't concentrate near us, stop here and report that one size is enough.

## Phase 4: four-turn lookahead (after Phase 2)

Keep the same single P (one next-turn predictor, applied again at each simulated state). Add a new network only if the offline error analysis below shows a systematic bias at some depth.

1. **Measure the current approximation first.** Today one second-turn prediction per sample is shared by every candidate plan, taken from the state the best plan reaches. Offline, on recorded states, compare it with a prediction per candidate. Measure the cost per turn and how often C08's chosen move changes.
2. **Give turns 3 and 4 their own simulated parents.** Each state is predicted with its immediate simulated parent as history. Extend the contract test to prove that branch order and count never change the real root or the next real turn.
3. **Re-derive the reach radius for each depth** from movement distance: 6 tiles at 2 turns; add the extra steps for 3 and 4.
4. **Make depth a setting from 2 to 4,** with every simulated turn settled by the exact kernel. Keep the engine comparison in the contract test.
5. **Run two modes:**
   - **fixed samples,** to isolate depth;
   - **fixed time,** because C08 stops its search on the wall clock.

   Compare against the 4-turn scripted control and the 2-turn P-only arm.
6. **Measure P's per-depth error** on coherent branch states labelled by the actual opponents' replies. Only then add training data from games against the P-only bot and fine-tune.
   - Never use P's own guesses as labels.
   - Keep each branch with its original map when splitting train and validation data.

## Phase 5: V, then confirmation

- **V:** after P is settled, compare at fixed P: V off, a simple score/time baseline, and the `_m_v` hybrid. Tune the coefficients on tuning maps only. Record how often V is actually used. V stays experimental until confirmed; its Brier score barely moves from S to XXL.
- **Freeze, then confirm:** freeze the finalists, including carrier, sizes, depth, coefficients and budgets. Confirm each with at least 10,000 independent map groups over counts 2–15 (at least 500 per count) on the unseen confirmation seed, with three independently trained seeds and paired fixtures.

## Evidence backfill for XL and XXL (GPU; doesn't block the screen)

The trend from S to XXL is one seed, 8 epochs, with no per-epoch logs (89ee436). These steps decide whether size conclusions can be trusted:
- **Correct the record:** 89ee436's commit message says `inv_teams` count weighting, but every manifest records `count_weighting none`. A correction note is now at the top of `reports_xl/SUMMARY.md` and `reports_xxl/SUMMARY.md`; don't rewrite the pushed history.
- **More seeds or the logs:** train 1–2 more seeds for L, XL and XXL with the identical configuration, or commit the per-epoch training logs, so we can tell whether XL and XXL were still improving at epoch 8.
- **Keep M separate:** M was trained with NumPy while L and up used PyTorch, so the M-to-L step mixes a backend change into the size change. Keep M in its own family and out of the size trend; compare sizes from L up, or retrain M on PyTorch.

## Report back

Each phase's results go under `p_only/report/` and are committed. Give a short summary with each:
- what was measured;
- the verdict per comparison;
- the timing, including the size rule's decision;
- what is still open.

If something is inconclusive, say so and keep the smaller or safer validated bot.
