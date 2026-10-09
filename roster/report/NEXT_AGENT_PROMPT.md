# Prompt for the next agent: finish the first roster evidence run and correct the report

Copy everything below the line into the next agent's first message.

---

You are taking over the opponent-roster work in **ctf-fusion** (`https://github.com/Casper015/ctf-fusion`, private, commits go directly to `main`) with **AI_Test** (`https://github.com/Casper015/AI_Test`) as the game engine source. Your job is to turn the first host run into **verifiable evidence** and a **correct report**, then complete the remaining first-run steps. This is an execution task, not a design essay. Write everything in English.

## 0. Read first, in this order

1. `docs/FIRST_RUN.md`: the runbook. Follow it; it is the protocol.
2. `roster/bots/README.md`: the 130 archived bot sources (122 eligible representatives), where they come from, how to run them.
3. `docs/ROSTER.md`: methods (evidence identity, distance v2, evidence states, live outcomes, registry, sampling).
4. `reports/FINGERPRINT_EFFECTIVENESS_REVIEW.md`: why the distance and evidence rules are what they are.
5. `roster/report/FINGERPRINT_BEHAVIOR_REPORT.md` and `roster/report/results/`: the first host run (commits `163322e`, `d665a0a`, `a72731a`). Read it critically; section 2 below lists its known problems.

Record the ctf-fusion and AI_Test commits you start from.

## 1. Hard rules

- **Host only.** Matches, fingerprints and latency run on the Windows host, never on a development laptop.
- **Never edit** `roster/bots/src/` (file names are checksums), a frozen probe suite after fingerprinting started, or any result file by hand. Results are produced by commands; reports are generated from result files by code.
- **Pre-registered settings.** Keep the defaults of `docs/FIRST_RUN.md` (6 maps per team count, `--repeat 2`, margins 0.02 / 0.05 / 0.05, z 1.645, at least 2 maps). If you must deviate, say so in the report with the reason.
- **No unsupported claims.** A fingerprint distance is a magnitude of disagreement on replayed histories. It has no direction. Do not call a bot aggressive, defensive, coordinated, base-camping or "Claude-like" unless a measured live metric with its denominator says so. Do not name MDS/PCoA axes; their orientation has no meaning by itself. Do not group bots by vendor or region.
- **Equivalence wording.** "Equivalent" means: within the pre-registered margins on this probe suite, upper bound over few maps (a heuristic, not a calibrated 95% guarantee). Never write "statistically identical", "zero difference" or "verifies that parameter changes did not alter behavior".
- **Report faithfully.** Include failures, faults, stale counts, unknown pairs and unmeasured bots. Do not rerun with other settings to improve a number.
- **Git.** Repositories outside OneDrive. Never bare `git stash`. Commit only relevant files with tests passing; end commit messages with your agent's attribution line. Do not commit `data/` wholesale (fingerprints are hundreds of MB); copy the listed small result files into `roster/report/results/` instead.

## 2. Known problems in the first host run (verified against its committed files)

1. **Protocol deviation.** It used 3 maps per team count (42 probes), not the runbook's 6 (84). With 3 maps many pairs should be *unknown*; the report never mentions unknown pairs, unmeasured bots or coverage.
2. **Which matrix?** `results/behavior_distance_matrix.csv` has no negative and no missing values, which points to the raw `overall` distance, while the report says "corrected for A/A noise" (`overall_excess` can be negative for stochastic bots). The exported field is not stated.
3. **Style labels contradict live data.** Seat outcomes recomputed from the 84 committed matches (714 seat-games; all equal the engine's own totals):

   | Bot | attack requests per shoot-capable unit-turn | kills per game | games |
   |---|---:|---:|---:|
   | Sonnet | 0.408 | 42.1 | 36 |
   | Sol | 0.376 | 39.8 | 40 |
   | Astra | 0.329 | 34.9 | 35 |
   | Opus | 0.237 | 27.1 | 26 |
   | GLM | 0.100 | 8.7 | 33 |

   Astra/Sol "aggressive" vs Sonnet/Opus "coordinated" does not hold. Opus, called an "aggressive skirmisher" in section 3.3, is in the lower half of the ten pool judge bots by attack request rate (0.237; Sonnet 0.408, Fable 0.241; Kimi, Qwen, MiMo, DeepSeek and GLM lower). The fingerprint-closest pair, Opus and Sonnet (0.203), differs strongly in live play. Doubao has no live data at all: it is not in `profiles/pool.json`. "Base defense" was never measured. These numbers are descriptive (mixed, unbalanced games), not conclusions.
4. **Overclaims.** "Zero statistical behavioral difference" for the 11 ablation variants, "statistically identical decisions", "verifying that parameter threshold shifts did not alter decision distributions", and the MDS axis names "search depth" and "skirmish posture".
5. **Evidence not committed.** `behavior_distance.json` (states, groups, coverage, stale runs), `probe_suite.json`, the host inventory and the assessments are missing. The 87 groups cannot be verified from the CSVs. The report cites `data/roster/all_bot_assessments.csv`, which is not in the repository.
6. **`matches_summary.csv` bug.** Empty `scores` and `winner_seat = -1` in all 84 rows, although every log has a score (the first is 26-21). The score is at `match["result"]["score"]`.
7. **Different bot id.** The host inventory chose `ctf-arena/Player7/workspace/experiments/round3/export/Player5_Omega_LIT.cpp` as representative; the committed archive inventory uses the byte-identical `ctf-benchmark-v3/track2_2k_c_series/bots/C08_Player5_Omega_LIT.cpp`. Same source hash, different id. Use `roster/bots/inventory.json` so ids join with the archive and the registry.
8. **Next-step recommendation.** Screening only the 87 group representatives is wrong for the open pool: probe equivalence is not equal strength, and all 122 cost about 40% more.

What did check out (keep it): no faults or dead runs in 10,248 replays; a consistent symmetric matrix; the judge table matches the CSV; S1_B_DuelPlus kept out of S1_A's group (its difference is only in 2-team duels); complete linkage split what a connected-component rule would have chained (16 bots at <= 0.02 on the averaged matrix).

## 3. Tasks, in order

### A. Setup and checks

- Fresh pull of both repositories. `python -m unittest discover -s tests` must end in `OK` with `AI_TEST_ROOT` set.
- `python -m roster.archive check --ai-test $AI_TEST_ROOT`: `130 stored files, 0 errors`.
- `python tools/env_check.py --ai-test $AI_TEST_ROOT --teams 2 6 15` (add `--bots` if a pool bot is missing). Keep its report.
- If the host's AI_Test has bots the archive lacks, run `python -m roster.inventory`, then `python -m roster.archive save --inventory data/roster/bot_inventory.json`, and commit the archive update first. Otherwise use `roster/bots/inventory.json` everywhere.

### B. The first full run, exactly as `docs/FIRST_RUN.md` section 5

Use new output folders (the 3-map fingerprints belong to another suite; the tools ignore them as stale, never mix them). Run the pilot (section 4) first and note its timings.

1. Probe histories from the archive pool, 12 matches per count, 14 counts (5.1).
2. Probe suite, 6 maps per count, 84 probes, no "fewer matches" warning (5.2).
3. Fingerprints of all 122 eligible bots, `--repeat 2` (5.3). A rerun must print `0 to do`.
4. Distances with `--inventory roster/bots/inventory.json` (5.4). Note how many bots are measured at every count and the stale-run count.
5. Strength schedule, screen of **all 122** with `--keep-matches`, aggregate with `--inventory` and `--schedule` (5.5-5.7).
6. Live profiles from the kept screen logs (5.8): `0 seat-games whose recomputed totals differ from the engine`.
7. Assessments and the admission template (5.9-5.10).

Record wall times of 5.1, 5.3, 5.6, the CPU and the worker counts.

### C. Commit the evidence

Copy into `roster/report/results/` (replace the 3-map files or keep them in `results/run1_3maps/`, labeled):

- `probe_suite.json`, `behavior_distance.json`, `strength.json`, `live_profiles.json`, `bot_assessments.json`, `all_bot_assessments.csv`, `admission.json`;
- `env_check_<host>.md/.json`;
- the probe-history matches that the suite references (as now under `results/matches/`), and `fixtures.jsonl`;
- `RUN.md`: commits, host, CPU, compiler, commands exactly as run, wall times, deviations.

Do not commit fingerprint runs or screen logs. Check the total added size (should be a few tens of MB at most).

### D. Generate the report from the files

Write `tools/generate_roster_report.py`. It reads only the committed result files and writes `roster/report/FINGERPRINT_BEHAVIOR_REPORT.md` and its figures. No number is typed by hand. Add a unit test that runs it on a small synthetic input (an existing distance test fixture is enough) and checks key lines. Remove or rewrite `tools/plot_style_space.py` accordingly. The report must contain:

1. **Run identity:** commits, suite identity, host, settings, deviations, counts of stale runs and stale strength rows.
2. **Reliability:** faults, dead runs, valid reply share per bot (fingerprints and screen), reply times labeled as relaxed-limit replay times (fingerprints) vs official-limit times (screen).
3. **Evidence states per team count:** pairs equivalent / different / unknown / not compared, bots measured, unmeasured bots listed.
4. **Groups:** per count and over all counts (bots measured everywhere), each with members and diameter; `established_distinct_counts`; the stochastic list. Wording as in section 1.
5. **Distance figures:** heatmaps of `overall_excess` (declare the field; if display clips negatives at 0, say so), judge-bot table with the same field. An MDS/PCoA figure is allowed only with unlabeled axes and a caption saying axes have no intrinsic meaning.
6. **Live outcomes per bot and team count** from `live_profiles.json` and the screen: attack request rate, hits per request, kills and deaths per game, death hazard, carrier share, delivery rate, threatened survival at 5 and 10 turns, each with numerator, denominator, games, unique maps and interval; censored counts where they apply.
7. **Strength:** utility and lower bound per team count and overall, maps, evidence level (screen or confirmation), faults, p99 under official limits.
8. **Descriptions, not labels:** any tendency statement cites the metric and its interval ("GLM's attack request rate 0.10 [interval] over N games"). No vendor groupings.
9. **Limits:** replay probes cannot steer private plans; few maps; threatened is a Manhattan proxy; live outcomes depend on the opponent mix.
10. **Next steps:** the admission review (a person fills `admission.json`), `roster.registry build`, confirmation runs for bots whose strength claims matter, frozen evaluation roster.

Also fix the summary script so `matches_summary.csv` carries real scores and winners (or drop the file in favor of the logs).

### E. Acceptance checklist (all must hold before you report done)

- [ ] Tests pass; `roster.archive check` has 0 errors.
- [ ] Suite has 84 probes (6 per count); fingerprints 122 bots x 84 x 2 with a `0 to do` rerun.
- [ ] `behavior_distance.json` committed; the report's group, unknown and unmeasured counts equal what the file contains (the test or a check script compares them).
- [ ] Strength screen covers all 122 bots at every team count; `stale_rows` reported.
- [ ] Live profiles show 0 engine mismatches; Doubao and every other eligible bot have live data from the screen.
- [ ] The report is generated, states which distance field each figure uses, and contains none of the overclaims in section 2.4 or any unmeasured style label.
- [ ] `matches_summary.csv` fixed or removed.
- [ ] `RUN.md` lists commits, settings, timings and deviations.

## 4. What to report back

A short message: what ran (with wall times), what failed or was skipped, the headline counts (measured bots, groups, unknown pairs per count, stale runs), where the files are, the commits you pushed, and what still needs a person (the admission review). Quote numbers from the files, not from memory.
