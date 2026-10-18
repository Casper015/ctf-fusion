# P-only paired screen

Schedule `66ca9a1b97a12d49a87a2657f193afcbdf882b7159e3035d2a07acb9af74ccd3`, seed 20262151, 784 fixtures over team counts 2-15 (56 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.394 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 3 comparisons (alpha 0.05): their intervals and verdicts use z = 2.394 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.394) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | all | -0.0040 | [-0.0230, +0.0151] | not shown | 784 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | 2 | -0.0268 | [-0.0978, +0.0442] | not shown | 56 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | 3-5 | -0.0211 | [-0.0735, +0.0313] | not shown | 168 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | 6-10 | +0.0136 | [-0.0164, +0.0436] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | 11-15 | -0.0067 | [-0.0340, +0.0206] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p2_gw8x4_fp32 | improved / unchanged / degraded | 257 / 261 / 266 | identical scores 1 of 784 | | |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | all | +0.0016 | [-0.0168, +0.0200] | not shown | 784 |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | 2 | +0.0179 | [-0.0566, +0.0924] | not shown | 56 |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | 3-5 | +0.0151 | [-0.0331, +0.0634] | not shown | 168 |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | 6-10 | +0.0050 | [-0.0234, +0.0335] | not shown | 280 |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | 11-15 | -0.0132 | [-0.0411, +0.0148] | not shown | 280 |
| agent_v2_p3exp_p3_direct - agent_v2_p3exp_p2_direct | improved / unchanged / degraded | 259 / 278 / 247 | identical scores 10 of 784 | | |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | all | -0.0197 | [-0.0390, -0.0004] | hurts | 784 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | 2 | -0.0804 | [-0.1976, +0.0369] | not shown | 56 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | 3-5 | -0.0684 | [-0.1233, -0.0136] | hurts | 168 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | 6-10 | -0.0003 | [-0.0293, +0.0286] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | 11-15 | +0.0024 | [-0.0189, +0.0236] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32 - agent_v2_p3exp_p3_direct | improved / unchanged / degraded | 192 / 386 / 206 | identical scores 155 of 784 | | |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | all | -0.0091 | [-0.0196, +0.0014] | not shown | 784 |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | 3-5 | -0.0196 | [-0.0517, +0.0125] | not shown | 168 |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | 6-10 | -0.0116 | [-0.0330, +0.0098] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | 11-15 | -0.0021 | [-0.0083, +0.0040] | not shown | 280 |
| agent_v2_p3exp_p3_gw8x4_fp32~aa - agent_v2_p3exp_p3_gw8x4_fp32 | improved / unchanged / degraded | 87 / 587 / 110 | identical scores 456 of 784 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_p3exp_p2_direct | 784 | 0.77332 | 0 | 0 | 0 | 2 | 2 | 0 | 52.4 | 1.092 | 6.186 | 15.768 | `3670d8907518` |
| agent_v2_p3exp_p3_direct | 784 | 0.77495 | 0 | 0 | 0 | 0 | 0 | 0 | 58.0 | 1.45 | 11.079 | 40.689 | `46c430100118` |
| agent_v2_p3exp_p2_gw8x4_fp32 | 784 | 0.75922 | 2 | 0 | 0 | 0 | 0 | 0 | 106.2 | 17.888 | 24.592 | 46.135 | `634d4d1b6a96` |
| agent_v2_p3exp_p3_gw8x4_fp32 | 784 | 0.75526 | 3 | 0 | 3 | 0 | 0 | 0 | 120.3 | 18.085 | 32.736 | 49.315 | `91c4cf029427` |
| agent_v2_p3exp_p3_gw8x4_fp32~aa | 784 | 0.74616 | 4 | 0 | 2 | 0 | 0 | 0 | 128.0 | 18.117 | 40.308 | 48.211 | `91c4cf029427` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
