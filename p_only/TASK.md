# Task for the host agent: P-only bots

Written 2026-10-09 for the agent that runs ctf-fusion on DESKTOP-G60OJUD. Read [README.md](README.md) first; it has the commands, what was built and what is already verified. Work in the phases below, in order, and commit each phase's results to `main` before starting the next.

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

## Phase 1: verify on the host

```sh
python -m unittest discover -s p_only/tests -v
python p_only/tools/build.py --stats
python p_only/tools/run_screen.py --smoke
```

All tests must pass on Windows. If the contract test fails, or a smoke game shows a fault, stop and report it with the output; don't screen.

Then measure timing under the screen's load: run the smoke at the official limits on more fixtures if needed (`--workers`), and record max reply, late replies and faults per arm.

**If C08 P-only M or L comes within 10 ms of the 50 ms limit,** reduce its ply-2 work before the screen. Reduce only that, and report the change:
- share the ply-2 view between samples whose ply-1 cells are equal;
- or predict ply 2 for fewer teams;
- or use a smaller policy size for ply 2 (see Phase 3).

## Phase 2: the paired screen

```sh
python p_only/tools/run_screen.py
```

The configuration is `p_only/profiles/screen.json`:
- 20 maps for each of the 14 team counts (2–15), with seed 20261031;
- 8 arms: control, earlier hybrid, P-only M and P-only L, for each carrier;
- two A/A copies of the controls;
- 2,800 games in total, an estimated 1–2 hours with 6 workers.

Keep the worker count fixed for the whole screen and record it. C08 searches on a wall-clock budget, so CPU load changes how it plays.

**Report, from `SCREEN.md`:**
- **Gain from P:** P-only versus control, per carrier and per band. This is the gain from replacing every hand-written prediction with P.
- **Pure versus mixed:** P-only versus the hybrid (does dropping the hand-written part help?), and L versus M.
- **C08 versus v12:** both P-only M and both P-only L.
- **Re-run noise:** the A/A rows (re-run noise and identical-score rate). Judge each C08 difference against them.
- **Timing and network calls:** faults, late replies, max reply, and network calls per turn.

Give a plain verdict for each comparison: helps, hurts or not shown.

**C08 diagnostics.** P-only also shortened its playouts from 4 plies to 2 and spends search time on network calls. If C08 P-only loses, separate the causes before concluding that P is worse:
- **Search samples:** compare the number of search samples per turn. `lastSearch.samples` is available in the planner; add it to the stats line if needed.
- **Deeper playouts:** test a P-only build with P-predicted plies 3–4 if the time budget allows.

## Phase 3: near opponents get a larger P, far ones a smaller P?

The original idea: in games with three or more teams, use a larger P for near opponents and a smaller P for distant ones. The P-only bots make this concrete.

**Measurements that frame the answer:**
- **Root cost:** all 14 opponents at 15 teams cost S 0.27 ms, M 0.34 ms, L 0.44 ms, XL 0.70 ms and XXL 1.34 ms per turn. Those are p99 figures with 16 seats running at once (`reports/capacity_DESKTOP-G60OJUD.md`).
- **So tiering saves almost nothing at the root:** everyone can already have the largest size that fits.
- **Inside C08 P-only's search it matters.** The smoke shows about 243 network views and 609 team encodes per turn, almost all of them for ply 2 of the sampled search, which only covers teams within 6 tiles of our units. A smaller P for those calls buys more samples within the same 50 ms. v12's ply 2 is much lighter (about 8 views per turn).
- **V can't be split by opponent:** V scores the whole state once per node. The V counterpart is a larger V near the root and a smaller one deeper in the search: a separate question.

**Steps:**
1. **Offline first:** do larger P models help more for near opponents?
   - On held-out records, compute each opponent team's NLL gain (L − S, and XXL − S once trained). Group it by wall-aware distance to our units, flags and carriers, by whether contact is possible within 2 turns, and by flag carrier.
   - On recorded states, swap far opponents' predictions between S and L and count how often the carrier's chosen orders change. Do the same for near opponents.
   - If the gains and order changes concentrate near us, tiering is promising. If not, use one size.
2. **Build the tiered arm only if step 1 supports it.** Root predictions stay at the large size. C08's ply-2 views use the large size for teams in contact (distance 3 or less, or a flag carrier) and the small size for the rest of the 6-tile radius. This needs two registries in one binary, compiled at the larger buffer width. Hold the per-turn wall-clock budget fixed so the saved time becomes more samples.
3. **Screen it:** compare against all-S, all-L and the largest size that fits, on fresh fixtures over counts 3–15, reported by band. Record which size served each prediction, the samples per turn, and the timing.

Expect no effect at the root, and a possible gain only where tiering buys more samples. Check this; don't assume it.

## Phase 4: V, then confirmation

- **V:** after P is settled, compare at fixed P: V off, a simple score/time baseline, and the `_m_v` hybrid. Tune the coefficients on tuning maps only. Record how often V is actually used. V stays experimental until it is confirmed.
- **Freeze, then confirm:** freeze the finalists, including carrier, sizes, coefficients and budgets. Confirm each with at least 10,000 independent map groups over counts 2–15 (at least 500 per count) on the unseen confirmation seed, with three independently trained seeds. Use paired fixtures.

## Report back

Each phase's results go under `p_only/report/` and are committed. Give a short summary with each:
- what was measured;
- the verdict per comparison;
- the timing;
- what is still open.

If something is inconclusive, say so and keep the smaller or safer validated bot.
