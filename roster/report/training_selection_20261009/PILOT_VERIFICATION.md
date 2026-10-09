# Pilot verification: 400 matches under `open_training_pool_recommended.json`

Pilot execution and validation on Windows host (`DESKTOP-G60OJUD`) on 2026-10-09.

This pilot verifies the recommended open training pool (`profiles/open_training_pool_recommended.json`, pool identity `85f96f37eeb7990645cacef17aee8b41a13b903db6e408a5da1d469388ec7383`) before proceeding to large-scale data collection.

## 1. Run configuration

- **Command:** `python -m training.collect --ai-test AI_TEST_ROOT --pool profiles/open_training_pool_recommended.json --teams 2 5 8 15 --matches 100 --seed 20261019 --out data/raw_recommended_pilot_20261009 --workers 8`
- **Team counts:** $N \in \{2, 5, 8, 15\}$, 100 matches per count (400 matches total, 3,000 seat-games).
- **Time limits:** Official competition limits (2.0 s turn 1, 50 ms later).
- **Random seed:** `20261019` (generating fresh, independent maps separate from earlier screen suites).
- **Wall time:** 232.0 s (~3.8 min) across 8 workers on AMD Ryzen 9 7945HX (average ~0.58 s per match).

## 2. Integrity and engine agreement

| Check | Result | Detail |
|---|---|---|
| **Fixtures on disk** | 400 / 400 | Exactly 100 gzip JSON files in each of `n02/`, `n05/`, `n08/`, `n15/` |
| **Independent maps** | 100% | 100 unique `(map_seed, flag_seed)` pairs per count |
| **Protocol faults** | **0** | No malformed replies, unhandled exceptions, or strike-outs |
| **Late replies** | **0** | All 55 profiled bots strictly met the 50 ms per-turn limit |
| **Engine agreement** | **0 mismatches** | `roster.trajectory` verified all 3,000 seat-games against `result.stats`; 0 discrepancies |
| **Combat symmetry** | Exact | Total fractional kills = 96,686.0; total deaths = 96,686.0 |
| **Total captures** | 23,404 | Objective progression confirmed active across all team counts |

## 3. Sampling and role composition

The hierarchical sampler matches the intended mixture parameters:

- **Field component:** 2,074 seats (**69.13%**, target 70%). Benchmark judge bots and selected challengers receive primary exposure.
- **Style component:** 918 seats (**30.60%**, target 30%). Behavior coverage across diverse lineages.
- **Augmentation regime:** 8 matches (**2.00%**, target 2% of fixtures with 1 augmentation seat).
- **Reserved OOD lineages:**
  - `lineage:AI_Arena_CTF_Player02_bot`: **0 seats**
  - `lineage:ctf-arena_Player4_v2`: **0 seats**
  Both evaluation lineages remain strictly unexposed.
- **Lineage concentration cap:**
  The largest lineage (`lineage:Player5`) is effectively bounded across counts:
  - $N=2$: 18 / 200 (9.00%)
  - $N=5$: 55 / 500 (11.00%)
  - $N=8$: 82 / 800 (10.25%)
  - $N=15$: 185 / 1500 (12.33%)

## 4. Voluntary drop and tactic coverage

Across the 400 pilot matches, **131 voluntary drop events** were recorded. They originate exclusively from the designated priority drop producers preserved during selection:

| Bot | Voluntary drops | Followed by friendly hit within 3 turns |
|---|---:|---:|
| `New_AI_Test/Player5/workspace/submission/bot.cpp` | 64 | 34 |
| `ctf-arena/Player7/judge/ctf/bots/fable-5.1.cpp` | 47 | 44 |
| `C04_G_PLAYOUT_4ply.cpp` | 13 | 9 |
| `ctf-stress-pack/bots_v2/Player5.cpp` | 4 | 2 |
| `C08_Player5_Omega_LIT.cpp` | 3 | 3 |
| **Total** | **131** | **92** |

## 5. Conclusion

The pilot demonstrates that `profiles/open_training_pool_recommended.json` runs cleanly, reliably, and without protocol faults under official competition time limits, while maintaining strict lineage caps, proper component mixture ratios, and steady generation of rare strategic behaviors (voluntary drops). The pool is ready for full-scale data collection.
