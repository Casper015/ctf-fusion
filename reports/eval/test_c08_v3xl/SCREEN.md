# P-only paired screen

Schedule `e23fa561705fd244519ace39448bea00afd8a23791dba0d3d76f88ecd0c6f28f`, seed 20261102, 70 fixtures over team counts 2-15 (5 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | all | +0.0286 | [-0.0314, +0.0885] | not shown | 70 |
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 5 |
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | 3-5 | -0.1306 | [-0.2948, +0.0337] | not shown | 15 |
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | 6-10 | +0.1510 | [+0.0314, +0.2706] | helps | 25 |
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | 11-15 | +0.0073 | [-0.0571, +0.0718] | not shown | 25 |
| c08_ponly_pxxl_v3xl_v - c08_ponly_xxl | improved / unchanged / degraded | 20 / 32 / 18 | identical scores 5 of 70 | | |
| c08_ponly_pxxl_v3xl_v - c08_control | all | +0.0336 | [-0.0280, +0.0953] | not shown | 70 |
| c08_ponly_pxxl_v3xl_v - c08_control | 2 | +0.1000 | [-0.0960, +0.2960] | not shown | 5 |
| c08_ponly_pxxl_v3xl_v - c08_control | 3-5 | -0.0611 | [-0.2658, +0.1436] | not shown | 15 |
| c08_ponly_pxxl_v3xl_v - c08_control | 6-10 | +0.0672 | [-0.0225, +0.1568] | not shown | 25 |
| c08_ponly_pxxl_v3xl_v - c08_control | 11-15 | +0.0437 | [-0.0279, +0.1152] | not shown | 25 |
| c08_ponly_pxxl_v3xl_v - c08_control | improved / unchanged / degraded | 30 / 24 / 16 | identical scores 0 of 70 | | |
| c08_ponly_xxl - c08_control | all | +0.0050 | [-0.0648, +0.0749] | not shown | 70 |
| c08_ponly_xxl - c08_control | 2 | +0.1000 | [-0.0960, +0.2960] | not shown | 5 |
| c08_ponly_xxl - c08_control | 3-5 | +0.0694 | [-0.0922, +0.2311] | not shown | 15 |
| c08_ponly_xxl - c08_control | 6-10 | -0.0838 | [-0.2315, +0.0639] | not shown | 25 |
| c08_ponly_xxl - c08_control | 11-15 | +0.0363 | [-0.0376, +0.1102] | not shown | 25 |
| c08_ponly_xxl - c08_control | improved / unchanged / degraded | 26 / 25 / 19 | identical scores 0 of 70 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c08_control | 70 | 0.73865 | 0 | 0 | 0 | 45.7 | 23.712 | 28.606 | 30.198 | `1ba9c44b889e` |
| c08_ponly_xxl | 70 | 0.7437 | 0 | 0 | 0 | 48.2 | 30.271 | 31.21 | 32.03 | `e12ca06aa829` |
| c08_ponly_pxxl_v3xl_v | 70 | 0.77227 | 0 | 0 | 0 | 49.5 | 30.349 | 31.189 | 31.554 | `e426690b1b39` |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c08_ponly_xxl | 28000 | 280.31 | 518.44 | 565.85 | 364.86 | 0 | 0 | 1113984 | 25935 | 358.0 |
| c08_control |  |  |  |  |  |  |  |  | 25935 | 512.0 |
| c08_ponly_pxxl_v3xl_v | 28000 | 279.24 | 512.63 | 552.63 | 357.1 | 0 | 0 | 1119213 | 25935 | 359.7 |

## Search work (C08, our seat)

Truncated: a deadline stopped the call before its requested samples. A fixed-sample arm (fixed > 0) is an equal-work comparison only for its untruncated calls.

| Arm | Fixed | Calls | Samples per call | Requested per call | Truncated share | Rollouts per call | Plans per call | Survivors per call | Lookahead cut share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c08_ponly_xxl | 0 | 25935 | 358.0 | 512.0 | 0.7031 | 3676.0 | 24.4 | 1.98 | 0.1107 |
| c08_control | 0 | 25935 | 512.0 | 512.0 | 0.0006 | n/a | 25.8 | n/a | n/a |
| c08_ponly_pxxl_v3xl_v | 0 | 25935 | 359.7 | 512.0 | 0.6957 | 3716.0 | 24.5 | 1.98 | 0.117 |

## Value term (V arms, our seat)

Changed: decisions where V picked differently from the same path without V. Zero calls on a path mean the path was not exercised, not that V has no effect there.

| Arm | Path | Coefficient | V calls per turn | Exact terminal | Decisions | Changed | Changed share | V ms per turn (all paths) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| c08_ponly_pxxl_v3xl_v | refine | 50.0 | 1.518 | 62 | 8164 | 1143 | 0.14 | 0.075 |
| c08_ponly_pxxl_v3xl_v | beam | 35.0 | 1.057 | 98 | 17295 | 2342 | 0.1354 | 0.075 |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
