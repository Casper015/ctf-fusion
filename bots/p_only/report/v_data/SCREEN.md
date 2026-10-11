# P-only paired screen

Schedule `4584cc6bd2a061f3f63f9d4a21a667377a69363eeb8518df8ed039b7b61ad6fd`, seed 20261201, 7000 fixtures over team counts 2-15 (500 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |
|---|---|---:|---|---|---:|

## Arms

| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v12_ponly_xxl | 7000 | 0.74672 | 0 | 0 | 0 | 60.2 | 5.641 | 12.428 | 15.621 | `6cbd5d34610f` |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v12_ponly_xxl | 2800000 | 7.66 | 57.12 | 246.16 | 84.29 | 0 | 0 | 0 |  |  |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
