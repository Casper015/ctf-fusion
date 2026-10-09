# Light strength screen

A small paired screen that separates performance tiers among the 64 bots of the draft admission (`roster/report/selection/admission_draft.json`) before any P/V data is collected from them. **Status: set up and checked with a 4-game smoke run; the screen itself has not run yet.** It runs on the host, because it plays real matches under the official limits.

## Run it (host)

```bash
python tools/run_light_screen.py
```

One command, configured by [`profiles/screen_light.json`](../../../profiles/screen_light.json). Every step resumes after an interruption. To check the plumbing first (4 games, 1 s turn limit, written under `data/roster/screen_light/smoke/`, not evidence):

```bash
python tools/run_light_screen.py --smoke
```

## Design

- **Candidates:** the bots admitted in the draft: 32 core and 19 distinct family representatives, 13 count specialists, Noise and Random.
- **Fixtures:** 5 maps per team count, 14 counts, so 70 games per candidate and 4,480 in all. Every candidate plays the same maps, seats and opponents (the frozen reference roster: ten judge bots, Player5, Baseline), so comparisons are paired. Seed `20261009`, separate from the probe maps.
- **Limits:** official (2 s first turn, 50 ms later); late replies and faults are recorded.
- **Expected precision:** pairwise utility per game has a spread of roughly 0.3, so 5 maps give about 0.13 SE per count and about 0.036 over the 14 counts, a 95% interval of about ±0.07. The screen separates bots whose average differs from parity by more than that; neighbors inside a tier are not ranked. These figures are an estimate; the screen reports its measured intervals.
- **Cost:** about 15-20 minutes at the earlier screen rate (about 4.6 games per second on the host); `RUN.json` records the real wall time.

## Tiers (pre-registered, `roster.tiers`)

| Tier | Rule |
|---|---|
| above | lower 95% bound of utility > 0.5: beats the reference field on average |
| level | the interval contains 0.5 |
| below | upper bound < 0.5 |

**Reliability.** A bot is unreliable at the official limits if it has any dead game, more than 0.01 faults per game, or more than 0.1 late replies per game.

**Screened admission.** The tiers feed a screened admission:
- unreliable bots are not admitted;
- reliable bots above the reference field join the field role (strong competition);
- judge bots keep the field role;
- weak bots stay in the pool, because diversity, not strength, is why they were picked.

`profiles/open_training_pool.json` is rebuilt from it.

## Files written here by the run

| File | Content |
|---|---|
| `strength.json` | utility per team count and overall, intervals, faults, reply times, live outcomes per candidate |
| `live_profiles.json` | live outcomes of every archived seat in the screen games (references included) |
| `tiers.json`, `TIERS.md` | ranked tiers, per-band utilities, reliability |
| `admission_screened.json` | the draft admission updated with screen results |
| `schedule.json` | the frozen fixtures |
| `RUN.json` | config, commits, host, wall time per step, games, tiers, unreliable bots |

Raw results and match logs stay in `data/roster/screen_light/` (not committed). After the run, commit `roster/report/screen_light/` and `profiles/open_training_pool.json`.
