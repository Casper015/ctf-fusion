# P-only paired screen

Schedule `fe686cacbaaf99ece5b3e497fd207b46ab46970e0f3cc2a23b52fb468f236a6e`, seed 20261955, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.5758 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 5 comparisons (alpha 0.05): their intervals and verdicts use z = 2.5758 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.5758) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | all | -0.0391 | [-0.0695, -0.0087] | hurts | 392 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | 2 | +0.0714 | [-0.0562, +0.1991] | not shown | 28 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | 3-5 | -0.0888 | [-0.1716, -0.0060] | hurts | 84 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | 6-10 | -0.0345 | [-0.0752, +0.0063] | not shown | 140 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | 11-15 | -0.0360 | [-0.0856, +0.0135] | not shown | 140 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large | improved / unchanged / degraded | 104 / 119 / 169 | identical scores 2 of 392 | | |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | all | -0.1476 | [-0.1812, -0.1141] | hurts | 392 |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | 2 | -0.0714 | [-0.2431, +0.1002] | not shown | 28 |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | 3-5 | -0.1696 | [-0.2515, -0.0878] | hurts | 84 |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | 6-10 | -0.1253 | [-0.1793, -0.0713] | hurts | 140 |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | 11-15 | -0.1719 | [-0.2200, -0.1238] | hurts | 140 |
| meow_rl_aligned_p2_large_r16 - meow_rl_aligned_p2_large | improved / unchanged / degraded | 64 / 89 / 239 | identical scores 0 of 392 | | |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | all | -0.0466 | [-0.0788, -0.0143] | hurts | 392 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | 2 | +0.0536 | [-0.0478, +0.1549] | not shown | 28 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | 3-5 | -0.0689 | [-0.1606, +0.0227] | not shown | 84 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | 6-10 | -0.0490 | [-0.0991, +0.0011] | not shown | 140 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | 11-15 | -0.0507 | [-0.0978, -0.0036] | hurts | 140 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large | improved / unchanged / degraded | 120 / 101 / 171 | identical scores 0 of 392 | | |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | all | +0.1010 | [+0.0655, +0.1366] | helps | 392 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | 2 | +0.1250 | [-0.0455, +0.2955] | not shown | 28 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | 3-5 | +0.1007 | [+0.0101, +0.1913] | helps | 84 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | 6-10 | +0.0763 | [+0.0167, +0.1359] | helps | 140 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | 11-15 | +0.1212 | [+0.0741, +0.1682] | helps | 140 |
| meow_rl_aligned_p2_large_w16 - meow_rl_aligned_p2_large_r16 | improved / unchanged / degraded | 202 / 79 / 111 | identical scores 0 of 392 | | |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | all | +0.1085 | [+0.0754, +0.1416] | helps | 392 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | 2 | +0.1429 | [-0.0175, +0.3032] | not shown | 28 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | 3-5 | +0.0809 | [-0.0098, +0.1715] | not shown | 84 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | 6-10 | +0.0909 | [+0.0446, +0.1371] | helps | 140 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | 11-15 | +0.1359 | [+0.0861, +0.1856] | helps | 140 |
| meow_rl_aligned_p2_large_fp32 - meow_rl_aligned_p2_large_r16 | improved / unchanged / degraded | 214 / 97 / 81 | identical scores 1 of 392 | | |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_aligned_p2_large_r16~aa - meow_rl_aligned_p2_large_r16 | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| meow_rl_aligned_p2_large | 392 | 0.76983 | 0 | 0 | 0 | 0 | 0 | 0 | 44.0 | 0.866 | 2.141 | 3.502 | `d76921a19675` |
| meow_rl_aligned_p2_large_fp32 | 392 | 0.73072 | 0 | 0 | 0 | 0 | 0 | 0 | 63.7 | 9.226 | 18.192 | 20.147 | `2b10a327ce7a` |
| meow_rl_aligned_p2_large_r16 | 392 | 0.62222 | 0 | 0 | 0 | 0 | 0 | 0 | 65.5 | 8.124 | 16.369 | 18.937 | `e7df2c2f043d` |
| meow_rl_aligned_p2_large_w16 | 392 | 0.72326 | 0 | 0 | 0 | 0 | 0 | 0 | 62.0 | 8.346 | 16.774 | 20.181 | `29eae4c39b6a` |
| meow_rl_aligned_p2_large_r16~aa | 392 | 0.62222 | 0 | 0 | 0 | 0 | 0 | 0 | 62.6 | 8.107 | 17.022 | 17.473 | `e7df2c2f043d` |

## Meowmind RL counters (our seat, the last cumulative MEOWRL line per game)

| Arm | Games | With counters | Turns counted / played | Refused | Search used | Search fallback | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large_r16 | 392 | 392 | 156800 / 156800 | 0 | 156800 | 0 | 1241006 |
| meow_rl_aligned_p2_large | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |
| meow_rl_aligned_p2_large_r16~aa | 392 | 392 | 156800 / 156800 | 0 | 156800 | 0 | 1241006 |
| meow_rl_aligned_p2_large_fp32 | 392 | 392 | 156800 / 156800 | 0 | 156800 | 0 | 1240100 |
| meow_rl_aligned_p2_large_w16 | 392 | 392 | 156800 / 156800 | 0 | 156800 | 0 | 1240022 |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large_r16 |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_large |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_large_r16~aa |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_large_fp32 |  |  |  |  |  |  |  |  |  |  |
| meow_rl_aligned_p2_large_w16 |  |  |  |  |  |  |  |  |  |  |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
