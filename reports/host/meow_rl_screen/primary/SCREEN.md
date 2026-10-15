# P-only paired screen

Schedule `8624b3ba1b81773d2a9a784fef92101b983a6188811815c89fa6a6c8554574f4`, seed 20261951, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | 95% interval | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| meow_rl_large_v_s101 - meow_rl_warm_start | all | +0.0318 | [+0.0080, +0.0557] | helps | 392 |
| meow_rl_large_v_s101 - meow_rl_warm_start | 2 | -0.0179 | [-0.0973, +0.0616] | not shown | 28 |
| meow_rl_large_v_s101 - meow_rl_warm_start | 3-5 | +0.0193 | [-0.0551, +0.0938] | not shown | 84 |
| meow_rl_large_v_s101 - meow_rl_warm_start | 6-10 | +0.0334 | [-0.0006, +0.0675] | not shown | 140 |
| meow_rl_large_v_s101 - meow_rl_warm_start | 11-15 | +0.0477 | [+0.0151, +0.0803] | helps | 140 |
| meow_rl_large_v_s101 - meow_rl_warm_start | improved / unchanged / degraded | 144 / 124 / 124 | identical scores 0 of 392 | | |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| meow_rl_large_v_s101~aa - meow_rl_large_v_s101 | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| meow_rl_large_v_s101 | 392 | 0.78698 | 0 | 0 | 0 | 51.7 | 0.738 | 2.657 | 4.551 | `5fee7b978189` |
| meow_rl_warm_start | 392 | 0.75514 | 0 | 0 | 0 | 49.5 | 0.739 | 2.893 | 4.873 | `a3b5e8e9f941` |
| meow_rl_large_v_s101~aa | 392 | 0.78698 | 0 | 0 | 0 | 62.4 | 0.731 | 3.097 | 7.358 | `5fee7b978189` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
