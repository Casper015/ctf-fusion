# First run: from a fresh checkout to the first roster evidence

This runbook is for anyone, person or AI agent, who runs ctf-fusion on the host for the first time. It takes the 130 archived bots ([roster/bots/README.md](../roster/bots/README.md)) through the first full evidence run: behavior fingerprints, strength screen, live outcomes and assessments. Run the steps in order. Each step names its command, how long it takes and what success looks like. Background and methods are in [ROSTER.md](ROSTER.md); this file only says what to do.

**Everything here was run end to end on a small scale:** the pilot below (2 team counts, 5 bots) took 77 s on the development Mac and gave the expected result. The full run has not been made yet; that is what this runbook is for. Its timings are estimates until the pilot runs on the host.

## 0. What you need

| Need | Detail |
|---|---|
| Machine | The Windows (or Linux) host with many cores that will collect data. Not a busy laptop: latency numbers only mean something on the host. |
| Python | 3.10 or newer, with numpy (`pip install -r requirements.txt`). PyTorch is not needed for this run. |
| C++ compiler | `g++` or `clang++` with C++20 on `PATH`. On Windows, [w64devkit](https://github.com/skeeto/w64devkit) unpacked at `C:/w64devkit` is found automatically, or set `CXX`. |
| AI_Test | A checkout of [AI_Test](https://github.com/Casper015/AI_Test) (the archive came from commit `c9c115b`). Only the engine (`ctf-arena/Player7/workspace/engine.py`) and the 12 reference bots of the strength screen come from it; every candidate bot comes from `roster/bots/`. |
| Disk | About 1 GB under `data/` for fingerprints and match logs. |
| Location | Both repositories **outside OneDrive** (OneDrive breaks git's file access). |

The commands below are for Git Bash (or any bash). In PowerShell set the variable with `$env:AI_TEST_ROOT = "C:/path/to/AI_Test"`; the `python` lines are the same.

## 1. Get the code (5 minutes)

```bash
git clone https://github.com/Casper015/ctf-fusion.git
```

```bash
cd ctf-fusion && pip install -r requirements.txt
```

```bash
export AI_TEST_ROOT=C:/path/to/AI_Test
```

Run every later command from the `ctf-fusion` folder, in the same shell (or set `AI_TEST_ROOT` again).

## 2. Check the setup (2 minutes)

```bash
python -m unittest discover -s tests
```

Success: the last line is `OK` (some tests say `skipped` without `AI_TEST_ROOT`; with it set, all run). Any `FAIL` or `ERROR`: stop and fix it first, nothing later is trustworthy.

```bash
python -m roster.archive check --ai-test $AI_TEST_ROOT
```

Success: `130 stored files, 0 errors`. The line about AI_Test files that changed since archiving is information only: the archive copy is the one that runs.

## 3. Host check (10-20 minutes, optional for this run, required before any latency claim)

```bash
python tools/host/env_check.py --ai-test $AI_TEST_ROOT --teams 2 6 15
```

It writes `reports/host/env_check_<host>.md`: compiler, CPU, protocol, bot latency under the real 50 ms limit, simulator exactness, SIMD kernel parity. Every verdict should pass. If it stops with `bot Player5_Omega_LIT not found`, that pool bot is missing from your AI_Test checkout: add `--bots Astra,Sol,Fable,Opus,Player5,Baseline` (any pool bots you have).

## 4. Pilot (2-5 minutes): prove the chain works on this machine

Use a separate folder so nothing mixes with the real run.

```bash
python -m roster.archive pool --out data/pilot/pool_archive.json
```

```bash
python -m training.collect --pool data/pilot/pool_archive.json --teams 3 6 --matches 3 --workers 2 --out data/pilot/raw_probes
```

```bash
python -m roster.probes --raw data/pilot/raw_probes --teams 3 6 --per-count 2 --turns 100 --out data/pilot/probe_suite.json
```

```bash
python -m roster.fingerprint --suite data/pilot/probe_suite.json --inventory roster/bots/inventory.json --only ctf-arena/Player7/judge/ctf/bots/gpt-6-astra.cpp ctf-arena/Player7/judge/ctf/bots/opus-5.5.cpp New_AI_Test/Player2/workspace/experiments/cand-002.cpp ctf-s1-controls/exports/S1_A_Conservative.cpp ctf-s1-controls/exports/S1_B_DuelPlus.cpp --repeat 2 --workers 2 --out data/pilot/fingerprints
```

```bash
python -m roster.distance --suite data/pilot/probe_suite.json --fingerprints data/pilot/fingerprints --inventory roster/bots/inventory.json --out data/pilot/behavior_distance.json
```

**Expected** (what the development machine gave):
- collect: `6/6` matches, `faults=0`;
- probes: `4 probes`;
- fingerprint: `5 bots (5 from the source archive) x 4 probes x 2 = 40 runs`;
- distance: `5 bots, 5 measured at every count, 4 behavior groups`, `0 stale runs`. S1_A_Conservative and S1_B_DuelPlus form one group at 3 and 6 teams (they are 99.96% the same code and differ only in 2-team duels); Astra, Opus and cand-002 are each their own group.

Note the fingerprint time: 40 runs of 100 turns. The full run is about 20,000 runs of 400 turns, so multiply accordingly to plan the night.

Run the pilot command again: it must print `0 to do` (finished runs are reused only when their evidence identity matches). Then delete `data/pilot`.

## 5. The first full run

Every step can be interrupted and re-run with the same command: finished work is skipped. Keep `--workers` at or below the physical cores (fingerprints: one bot per worker; strength: a whole match per worker, so workers x team count should stay near the core count).

| Step | Command | Estimate | Success |
|---|---|---|---|
| 5.1 Probe histories | see below | 30-60 min | `168/168` matches (12 per team count) |
| 5.2 Probe suite | see below | 2 min | `84 probes`, no "fewer matches" warning |
| 5.3 Fingerprints | see below | 4-8 h on 6 workers | ends without a long list of faults; re-running prints `0 to do` |
| 5.4 Distances | see below | 10-30 min | most of the 122 bots `measured at every count`, `0 stale runs` |
| 5.5 Strength schedule | see below | 1 min | `700 fixtures` |
| 5.6 Strength screen | see below | about 5 h | every candidate played every fixture |
| 5.7 Strength aggregate | see below | 1 min | `0 stale rows` |
| 5.8 Live profiles | see below | 10-20 min | `0 seat-games whose recomputed totals differ from the engine` |
| 5.9 Assessments | see below | 1 min | `122` eligible rows with `fingerprinted`/`screened` status |
| 5.10 Admission template | see below | 1 min | `data/roster/admission.json` with nothing admitted |

5.1 Record diverse probe histories from the archived bots (12 matches per team count, so 6 distinct maps per count survive the fault filter):

```bash
python -m roster.archive pool --out profiles/pool_archive.json
```

```bash
python -m training.collect --pool profiles/pool_archive.json --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --matches 12 --workers 8 --out data/raw_probes
```

5.2 Freeze the probe suite (6 maps per team count):

```bash
python -m roster.probes --raw data/raw_probes --per-count 6 --out data/roster/probe_suite.json
```

If it warns `fewer matches than requested`, re-run 5.1 with `--matches 16` (the finished matches are kept) and repeat 5.2. Do not change the suite after 5.3 starts: a re-frozen suite makes every fingerprint stale.

5.3 Fingerprint all 122 eligible bots, two repeats each:

```bash
python -m roster.fingerprint --suite data/roster/probe_suite.json --inventory roster/bots/inventory.json --repeat 2 --workers 6
```

5.4 Distances, evidence states and coverage:

```bash
python -m roster.distance --suite data/roster/probe_suite.json --inventory roster/bots/inventory.json --out data/roster/behavior_distance.json
```

5.5-5.7 Strength screen against the frozen reference roster (10 judge bots, Player5, Baseline), 50 maps per team count, keeping full logs for live outcomes:

```bash
python -m roster.strength schedule --maps-per-count 50 --out data/roster/screen_schedule.json
```

```bash
python -m roster.strength run --schedule data/roster/screen_schedule.json --inventory roster/bots/inventory.json --workers 8 --keep-matches data/raw_screen
```

```bash
python -m roster.strength aggregate --results data/roster/screen_results.jsonl --inventory roster/bots/inventory.json --schedule data/roster/screen_schedule.json --out data/roster/strength.json
```

5.8 Live outcome profiles from the kept logs (every seat whose source is archived, reference bots included):

```bash
python -m roster.trajectory --raw data/raw_screen --inventory roster/bots/inventory.json --out data/roster/live_profiles.json
```

5.9-5.10 One row per candidate, then the admission file for a person to fill in:

```bash
python -m roster.assess --inventory roster/bots/inventory.json --suite data/roster/probe_suite.json --distance data/roster/behavior_distance.json --strength data/roster/strength.json
```

```bash
python -m roster.registry template --assessments data/roster/bot_assessments.json --distance data/roster/behavior_distance.json --out data/roster/admission.json
```

## 6. What to send back

`data/` is not committed (fingerprints and logs are large). Share these small files, for example on a branch `results/first-run-<host>` with `git add -f`:

| File | What it holds |
|---|---|
| `reports/host/env_check_<host>.md` and `.json` | Host, compiler, latency baseline (step 3) |
| `data/roster/probe_suite.json` | The frozen probes (identity of every fingerprint) |
| `data/roster/behavior_distance.json` | Evidence states, groups, coverage, stale runs |
| `data/roster/strength.json` | Utility per team count, faults, reply times, live outcomes per candidate |
| `data/roster/live_profiles.json` | Live outcomes of every archived bot seen in the screen |
| `data/roster/all_bot_assessments.csv` and `bot_assessments.json` | One row per candidate |
| `data/roster/admission.json` | The template to fill in |

Also report: the commit of ctf-fusion and of AI_Test you ran, the CPU, the wall time of 5.1, 5.3 and 5.6, and any step that printed faults, warnings or stale counts. Report failures as they are; do not re-run a step with different settings to make a number look better.

## 7. If something goes wrong

| Symptom | Cause and fix |
|---|---|
| `no C++ compiler found` | Install w64devkit at `C:/w64devkit`, or set `CXX` to your `g++.exe`. |
| `sources changed since the inventory` | A bot file differs from its archived hash. With the default `--source archive` this means `roster/bots/src` was edited: restore it with `git checkout roster/bots/src`. |
| `bot X not found at ...` during collect | A pool entry is missing from your AI_Test checkout. Use the archive pool (`profiles/pool_archive.json`), as in 5.1. |
| `fewer matches than requested` | Collect more matches (5.1 with a larger `--matches`), then re-freeze the suite before any fingerprinting. |
| Many `stale runs` in 5.4 | Fingerprints of another suite or older sources are in `data/roster/fingerprints`. They are ignored, never mixed; delete them if space matters. |
| A bot shows many faults or `dead` runs | It crashes or answers malformed lines. It stays in the inventory but cannot be admitted (reliability gate). Report it. |
| A step was interrupted | Run the same command again. |
| `results mix several source hashes` | Pass `--inventory roster/bots/inventory.json` to `strength aggregate`. |

## 8. Rules for an AI agent running this

- Run on the host only. Do not run latency tests or large match batches on a development laptop.
- Never edit `roster/bots/src/`, the suite after fingerprinting started, or a result file by hand.
- Change no thresholds, margins or defaults during the first run; they are pre-registered. Record any deviation.
- Report what the commands printed, including failures and skipped steps. A synthetic or pilot result is not a roster result.
- Commit code changes only with their tests passing; results go on a separate results branch.
- Keep the repositories outside OneDrive. Never use bare `git stash` in a shared worktree.

## After the first run

A person fills in `data/roster/admission.json`: admit bots with one sentence each on what they contribute, reserve whole held-out lineages, and choose roles. Then follow [ROSTER.md](ROSTER.md) *Host command sequence* from `roster.registry build`: the open training pool, collection from it, confirmation of the bots whose strength claims matter, and the frozen evaluation roster.
