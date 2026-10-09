# Independent review of the committed pilot evidence

Reviewed source: Casper015/ctf-fusion at `9ee676b`. Controlling input-byte hashes are recorded in [pilot_aggregate_audit.json](pilot_aggregate_audit.json). Review date: 2026-10-09.

## Decision

The published pilot evidence supports continuing **staged** collection with the recommended pool. Keep the membership and sampling settings. It demonstrates useful event generation and internally consistent published aggregates. It does not establish sufficient training data, optimal opponent weights, successful flag tactics, or an independently reproduced timing certificate.

The raw 400 match logs, fixture manifest and final host verifier are not included in this commit. This review checks the published summaries and profiles and reproduces the deterministic fixture plan; it does not rerun the bots, replay the raw games, train a model, or benchmark latency. The output explicitly separates independently checked aggregates from producer-reported fault/late counts.

## What was independently checked

| Item | Result |
|---|---|
| Pool identity | Recomputed from the pool's controlling membership/settings; matches `85f96f37...ec7383` |
| Fixture plan | 400 matches, 3,000 seat-games, 55 observed programs; matches the reported seed and configuration |
| Bot/count exposure | Every published bot/count game denominator exactly matches the reproduced seat schedule |
| Components | 2,074 field, 918 style, 8 augmentation seats; matches the summary |
| Augmentation | 392 mixed fixtures, 8 augmented fixtures; 2% of fixtures, not 2% of seats |
| Map/flag seeds | The reproduced plan has 100 unique map seeds and 100 unique flag seeds at each selected count |
| Reserved lineages | Zero planned/profiled appearances of both reserved families |
| Core event aggregates | Published profiles match captures, drops and kill-credit totals at the summary's displayed precision |
| Engine-mismatch counter | All published bot/count counters sum to zero |
| Drop sources | Exactly the five reported sources; all per-source drop/follow-up totals match |

Official judge bots occupy 1,686 / 3,000 seats, or **56.2%**, close to the proposal's sampler estimate. This is a useful confirmation that the composition goal survives collection. It is not a strength improvement measurement.

Fractional kills summed from the rounded profile entries are `96685.9906`, versus `96686` deaths; carrier-kill credits are `24829.9931`. They round to the reported `96686.0` and `24830.0`. Call this agreement at reported precision, rather than literal exact floating-point equality. No meaningful event-count discrepancy was established.

## Limits to the report's interpretation

**Timing.** The public summary records `protocol_faults=0`, `late_replies=0` and official limit settings. It publishes no per-bot/count `first_ms`, `p99_ms`, `max_ms`, timed-turn count or raw timing records. These zeros remain the host producer's claim; the committed material alone does not let this reviewer independently recheck them. The earlier pasted verifier used nonexistent report keys `late_replies` and `max_turn_ms`; the real `fusion.runner.Bot.report()` keys are `late` and `max_ms`. The final edited host script was not provided, so this is a required check, not proof that its final numbers are wrong.

Require telemetry keys rather than using `.get(missing_key, 0)`. Treat absent/null timings as unknown, not zero. Check first-turn and later limits separately; successful later-response percentiles exclude timeouts, so retain faults, dead processes and timed-turn counts too. Count the match fault list once, and reconcile it with per-bot fault totals instead of adding both views. Archive the final verifier, run/engine/source identities and per-file checksums; raw logs can stay on the host.

**Soft caps.** The 10% setting is an initial per-component lineage cap. Sampling without replacement changes final seat shares; 11% or 12.33% is not failure of that declared rule. The report should say concentration was reduced and actual shares checked, not that a strict 10% final-seat ceiling was enforced.

**Drop interpretation.** A friendly hit within three turns can be made by another unit and may have occurred without dropping. The 92 / 131 follow-up count is not a handoff success rate, a dropping-unit attack rate, or a causal benefit estimate. Prediction training can still use these valid requested behaviors even when a tactic is ineffective; P predicts opponents rather than imitating an optimal strategy. Successful handoff/delivery needs flag-aware episode evidence and appropriate comparison contexts.

**Coverage.** The 131 drops split as follows:

| Teams | Drops |
|---:|---:|
| 2 | 11 |
| 5 | 55 |
| 8 | 10 |
| 15 | 55 |

The pilot has up to 1.2 million raw team-turn records at 400 turns/game, so drop events amount to only about **0.011%** of those rows before subsampling. This is not the appropriate denominator for a carrier's conditional drop probability, but it exposes overall class scarcity. Multiple drops can share a game or team-turn. Published `unique_maps` counts measure metric exposure, not distinct maps containing a drop. Report event-containing maps and critical windows from raw logs before claiming broad rare-tactic coverage.

Two retained drop producers are lightly exposed: `bots_v2/Player5.cpp` played 10 seat-games and Omega_LIT 14, producing 4 and 3 drops. All drops come from five policies, and two supply 111 / 131 events. The fact that a behavior was captured is useful; it does not prove stable coverage of its many tactical conditions. Preserve drop windows when building shards: the global stride can otherwise discard the key turn even when flag/contact retention is 1.

**Count and program coverage.** Counts 3, 4, 6, 7, 9, 10, 11, 12, 13 and 14 are untested by this pilot. `cand-mc2` and `S1_C` were inactive at the four pilot counts; archive HABITS was allowed at 2 but happened never to be sampled. Thus 55 cleanly profiled programs is not evidence for every retained program/count combination. Earlier strength-screen evidence remains relevant, but is a different experiment.

The previous carrier-episode ordering and fingerprint dependency/limit defects were repaired in `3a1110a`, before this pilot evidence commit. Do not keep listing them as unfixed. Unit-test success supports those code repairs; it does not itself validate the private pilot corpus. Seventeen skipped tests are not seventeen completed checks.

## Next phase on the designated host

First fill missing count coverage, using a new directory and the same frozen pool:

```sh
python -m training.collect --ai-test AI_TEST_ROOT --pool profiles/open_training_pool_recommended.json --teams 3 4 6 7 9 10 11 12 13 14 --matches 100 --seed 20261020 --out data/raw_recommended_count_smoke_20261009 --workers 8
```

This is 1,000 additional matches. Also check per-source exposure so permitted but unsampled representatives receive a legal controlled fixture if necessary. Preserve their existing count-admission rules.

Then collect the first normal-distribution training phase at **1,500 matches for each count 2-15**, rather than immediately assuming 5,000/count is sufficient or necessary:

```sh
python -m training.collect --ai-test AI_TEST_ROOT --pool profiles/open_training_pool_recommended.json --teams 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --matches 1500 --seed 20261029 --out data/raw_train_v1_20261009 --workers 8
```

This is 21,000 matches. Replace `AI_TEST_ROOT` with the checkout path. These commands were prepared, **not run here**. Freeze the pool, sources, engine, collection limits and split policy; verify the new directories are disjoint from selection/evaluation maps. Use an immutable run manifest and refuse incompatible resume rather than accepting any existing filename.

The pilot's 232s/400 figure is throughput of about 1.72 matches/s, not single-match latency. Linear extrapolation gives about 3.4 hours for 21,000 matches, excluding verification/extraction/training; count mix, contention and I/O can change it. It is a planning estimate, not a runtime promise.

Build shards and run a small end-to-end training/calibration/integer-export pilot before spending the entire corpus budget. Start with S/M general models and representative dedicated counts; verify losses, legal-label masks, augmentation exclusion, exact inference and artifact gates. Inspect event counts **after** subsampling and split filtering. Keep a natural-distribution evaluation set alongside targeted rare-event windows; do not mix oversampled windows into headline metrics without declared weights.

Increase to 3,000-5,000 matches/count only if independent-map learning curves or tactical coverage justify it. Compare model capacities on identical map subsets, optimizer budgets and untouched evaluation maps/lineages. V targets must name their continuation mixture. Neither zero faults nor more rows demonstrates prediction gain or equal-time playing-strength gain.

To repeat this public aggregate review:

```sh
python3 tools/check_pilot_evidence.py --out roster/report/training_selection_20261009/pilot_aggregate_audit.json
```

The output deliberately sets `raw_logs_replayed=false` and `raw_timing_rechecked=false`. Do not promote it to a full raw-data or timing certificate.
