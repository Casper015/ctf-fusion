# P-only paired screen

Schedule `c7ab9ca4dfff5db5a5d5000ae36f65e7fdcf850d367910b20907a2c4a34c2301`, seed 20261954, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | Interval (95%) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | all | +0.0384 | [+0.0166, +0.0601] | helps | 392 |
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | 2 | +0.0357 | [-0.0343, +0.1057] | not shown | 28 |
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | 3-5 | +0.0600 | [-0.0003, +0.1204] | not shown | 84 |
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | 6-10 | +0.0289 | [-0.0069, +0.0648] | not shown | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | 11-15 | +0.0354 | [+0.0050, +0.0657] | helps | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_warm | improved / unchanged / degraded | 156 / 128 / 108 | identical scores 0 of 392 | | |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | all | +0.0117 | [-0.0077, +0.0311] | not shown | 392 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | 3-5 | +0.0407 | [-0.0113, +0.0926] | not shown | 84 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | 6-10 | -0.0047 | [-0.0410, +0.0316] | not shown | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | 11-15 | +0.0131 | [-0.0128, +0.0389] | not shown | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p1_large | improved / unchanged / degraded | 139 / 130 / 123 | identical scores 0 of 392 | | |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | all | -0.0059 | [-0.0254, +0.0137] | not shown | 392 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | 2 | +0.0357 | [-0.0343, +0.1057] | not shown | 28 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | 3-5 | +0.0134 | [-0.0367, +0.0635] | not shown | 84 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | 6-10 | -0.0246 | [-0.0602, +0.0110] | not shown | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | 11-15 | -0.0070 | [-0.0320, +0.0180] | not shown | 140 |
| meow_rl_aligned_p2_large - meow_rl_aligned_p2_small | improved / unchanged / degraded | 119 / 128 / 145 | identical scores 1 of 392 | | |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_aligned_p2_large~aa - meow_rl_aligned_p2_large | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| meow_rl_aligned_p2_large | 392 | 0.78471 | 0 | 0 | 0 | 0 | 0 | 0 | 75.2 | 0.947 | 4.335 | 7.328 | `d76921a19675` |
| meow_rl_aligned_warm | 392 | 0.74633 | 0 | 0 | 0 | 0 | 0 | 0 | 67.8 | 0.938 | 4.124 | 15.811 | `4538f0d302bd` |
| meow_rl_aligned_p1_large | 392 | 0.77301 | 0 | 0 | 0 | 0 | 0 | 0 | 107.3 | 0.95 | 3.885 | 31.804 | `1ac91250e5ab` |
| meow_rl_aligned_p2_small | 392 | 0.79058 | 0 | 0 | 0 | 0 | 0 | 0 | 92.3 | 0.919 | 4.625 | 16.167 | `59d8aade3d63` |
| meow_rl_aligned_p2_large~aa | 392 | 0.78471 | 0 | 0 | 0 | 0 | 0 | 0 | 99.9 | 0.92 | 3.882 | 15.035 | `d76921a19675` |

## Meowmind RL counters (our seat, the last cumulative MEOWRL line per game)

| Arm | Games | With counters | Turns counted / played | Refused | Search used | Search fallback | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |
| meow_rl_aligned_warm | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |
| meow_rl_aligned_p2_large~aa | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |
| meow_rl_aligned_p2_small | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |
| meow_rl_aligned_p1_large | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_warm |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_large~aa |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_small |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p1_large |  |  |  |  |  |  |  |  |  |  |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
