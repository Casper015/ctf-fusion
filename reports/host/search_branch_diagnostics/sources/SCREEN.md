# P-only paired screen

Schedule `5135e7483b6aa2bb196e162bb9eafb900e9e82777091397e6a4e813de5477768`, seed 20262160, 112 fixtures over team counts 2-15 (8 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | Interval (95%) | Verdict | Fixtures |
|---|---|---:|---|---|---:|

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_p3exp_p2_direct | 112 | 0.68149 | 0 | 0 | 0 | 0 | 0 | 0 | 48.6 | 0.943 | 3.314 | 6.146 | `3670d8907518` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
