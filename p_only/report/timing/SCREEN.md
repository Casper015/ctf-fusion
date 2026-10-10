# P-only paired screen

Schedule `bb6e85e6d52200b4b844f811259359a831e36207790f6c2785b1e479dee84b81`, seed 20261101, 15 fixtures over team counts 8-15 (5 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |
|---|---|---:|---|---|---:|

## Arms

| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c08_control | 15 | 0.58506 | 0 | 0 | 0 | 42.0 | 25.833 | 28.362 | 29.367 | `cc509993a3be` |
| c08_ponly_m | 15 | 0.69242 | 0 | 0 | 0 | 45.9 | 30.165 | 30.199 | 30.255 | `04b2ab3e4d46` |
| c08_ponly_l | 15 | 0.72554 | 0 | 0 | 0 | 44.0 | 30.174 | 30.211 | 30.737 | `229b1e6bdd00` |
| c08_ponly_xl | 15 | 0.72641 | 0 | 0 | 0 | 40.2 | 30.22 | 30.597 | 31.017 | `8f08a1dc0d55` |
| c08_ponly_xxl | 15 | 0.77511 | 0 | 0 | 0 | 42.6 | 30.652 | 31.198 | 31.442 | `bbbb1d1b1199` |
| v12_ponly_xl | 15 | 0.7658 | 0 | 0 | 0 | 46.7 | 5.796 | 8.053 | 8.91 | `0e8b839ce380` |
| v12_ponly_xxl | 15 | 0.73225 | 0 | 0 | 0 | 50.5 | 8.453 | 10.55 | 12.165 | `9898cb04d6c0` |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c08_ponly_m | 6000 | 439.45 | 980.03 | 969.86 | 679.04 | 0 | 0 | 421485 | 5985 | 492.8 |
| c08_ponly_xxl | 6000 | 275.37 | 536.02 | 641.76 | 382.89 | 0 | 0 | 254104 | 5985 | 327.3 |
| c08_ponly_xl | 6000 | 377.94 | 803.0 | 855.85 | 565.79 | 0 | 0 | 343066 | 5985 | 427.7 |
| c08_ponly_l | 6000 | 431.48 | 945.5 | 951.11 | 649.91 | 0 | 0 | 386345 | 5985 | 477.3 |
| v12_ponly_xl | 6000 | 7.85 | 82.37 | 340.55 | 108.91 | 0 | 0 | 0 |  |  |
| c08_control |  |  |  |  |  |  |  |  | 5985 | 511.9 |
| v12_ponly_xxl | 6000 | 7.69 | 80.63 | 334.96 | 105.74 | 0 | 0 | 0 |  |  |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
