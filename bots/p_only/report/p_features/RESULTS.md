# T0013 P2 versus P3 at equal full-turn limits

At the fixed 1,960-game tuning-screen endpoint, P3 playing-strength improvement is not shown for either carrier or any prespecified band. T0011's prediction gains do not translate into a demonstrated advantage here. Production defaults remain unchanged; this screen alone does not select P3 for confirmation.

| Carrier: P3 minus P2 | Mean utility difference | 95% paired map interval | Verdict |
|---|---:|---|---|
| v12 | +0.01102 | [-0.01761, +0.03965] | not shown |
| c08 | +0.00209 | [-0.02559, +0.02977] | not shown |

Utility is the fraction of opponents beaten, with ties worth one half. Counts 2–15 receive equal weight. The unit of pairing is one of 280 independent map fixtures; seven arms share map/flag/seat/opponents. All original-reference comparisons and the carrier interaction also contain zero. The complete per-count/band table and timing/counters are in [SCREEN.md](SCREEN.md) and [screen.json](screen.json). A/A has identical final score vectors for all 280 maps and zero utility difference; this supports the screen check, not universal freedom from scheduling effects.

The bounded design is 14 counts x 20 maps x (six arms + one A/A copy) = 1,960 games, master 20261210, V off, 2,000 ms first reply and 50 ms subsequent replies, six workers on DESKTOP-G60OJUD under one exclusive CPU lease. No extra seed, favorable interim stopping or endpoint expansion. All arms use unchanged T0011 seed1 exports; all four rebuilt P2/P3 carrier executables have exactly the prior verified T0011 binary hashes. Carrier-header commentary and consequently bundled source hashes differ; current full source identities are frozen.

Every game and all 784,000 turn blocks are retained. The independent audit verified 6,664,000 canonical all-seat turn replies, zero missing/invalid replies, zero all-seat faults, zero candidate late replies, raw/binary/source/seat/map/flag/limits identities and all six paired/interaction/A/A comparisons. Raw receipts are in [RAW_RECEIPTS.json](RAW_RECEIPTS.json). Success reply timing fields were matched to raw reports; individual samples are not retained for independently recomputing percentiles.

The measured P3 cost is higher. The policy-only values below come from recorded n02/n04/n15 streams with actual weights; each entry is the worst copy's p99 in microseconds. The original value-bench tool also exercises the shared V/M subnetwork, although V remains off in games. Cost probe times and the V-only fields are not full-turn P deployment latency.

| Concurrent copies | P2 view / encode / joint p99 us | P3 view / encode / joint p99 us |
|---:|---|---|
| 1 | 2.0 / 2.0 / 35.0 | 4.0 / 4.0 / 64.0 |
| 6 | 14.0 / 4.0 / 37.0 | 21.0 / 7.0 / 49.0 |
| 15 | 19.0 / 6.0 / 57.0 | 30.0 / 11.0 / 88.0 |

C08 P2 averaged 360.1 samples per search call and P3 276.0 (both requested 512). Truncated-call shares were 0.6996 and 0.8384, respectively. SEARCH counters are retained original analysis; the independent auditor independently recomputed PONLY totals, but not the SEARCH ratios. The cost/run difference is descriptive for this host run, not a same-work or causal allocation experiment. Flag-return skipped states are not excluded faults or missing replies.

Seed preparation checked the original ID index, all 50,000 actual T0028 records and inspected previous schedule files; no map intersection or prior formal schedule with this master was found. This is limited to those recorded inputs, not a universal historical-map claim. The seed is now consumed.

No game replay, retraining, model export or default/routing change. T0032 is a separate future candidate set and game budget; its offline teacher is not deployed. T0013 ends at this endpoint. See [INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json) for audit limits and [REPRODUCE.md](REPRODUCE.md) for commands and exact helper source.
