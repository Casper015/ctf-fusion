# P-only paired screen

Schedule `469de85a9e6903b3af494bda17663f60f108525ef7f8170f0e3a645c66da6968`, seed 20261801, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | all | -0.1828 | [-0.2114, -0.1543] | hurts | 392 |
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | 2 | -0.3036 | [-0.4657, -0.1415] | hurts | 28 |
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | 3-5 | -0.1860 | [-0.2606, -0.1114] | hurts | 84 |
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | 6-10 | -0.1589 | [-0.2008, -0.1171] | hurts | 140 |
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | 11-15 | -0.1807 | [-0.2203, -0.1410] | hurts | 140 |
| v12_ponly_xxl_first_trial_mlp_s101 - v12_ponly_xxl | improved / unchanged / degraded | 56 / 95 / 241 | identical scores 0 of 392 | | |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | all | -0.1555 | [-0.1848, -0.1261] | hurts | 392 |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | 2 | -0.2321 | [-0.3874, -0.0769] | hurts | 28 |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | 3-5 | -0.1493 | [-0.2300, -0.0686] | hurts | 84 |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | 6-10 | -0.1209 | [-0.1649, -0.0770] | hurts | 140 |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | 11-15 | -0.1783 | [-0.2173, -0.1393] | hurts | 140 |
| v12_ponly_xxl_first_trial_linear_s101 - v12_ponly_xxl | improved / unchanged / degraded | 69 / 103 / 220 | identical scores 0 of 392 | | |
| v12_ponly_xxl~aa - v12_ponly_xxl | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| v12_ponly_xxl~aa - v12_ponly_xxl | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| v12_ponly_xxl~aa - v12_ponly_xxl | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| v12_ponly_xxl~aa - v12_ponly_xxl | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| v12_ponly_xxl~aa - v12_ponly_xxl | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| v12_ponly_xxl~aa - v12_ponly_xxl | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v12_ponly_xxl | 392 | 0.74574 | 0 | 0 | 0 | 47.2 | 4.864 | 9.682 | 11.316 | `d8095b80b5bd` |
| v12_ponly_xxl_first_trial_linear_s101 | 392 | 0.59029 | 0 | 0 | 0 | 46.9 | 5.044 | 9.906 | 11.519 | `08df6938190c` |
| v12_ponly_xxl_first_trial_mlp_s101 | 392 | 0.5629 | 0 | 0 | 0 | 45.5 | 4.91 | 9.916 | 12.045 | `4409462c5012` |
| v12_ponly_xxl~aa | 392 | 0.74574 | 0 | 0 | 0 | 49.2 | 4.952 | 9.633 | 10.856 | `d8095b80b5bd` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
