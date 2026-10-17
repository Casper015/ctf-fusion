# P-only paired screen

Schedule `171daa9cde0f9b2b3a12e0d3d15a9da4f4090a309bf80b9ae42588fb6dd4a78a`, seed 20262131, 1344 fixtures over team counts 2-15 (96 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.2414 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 2 comparisons (alpha 0.05): their intervals and verdicts use z = 2.2414 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.2414) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | all | -0.0310 | [-0.0447, -0.0174] | hurts | 1344 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | 2 | -0.0469 | [-0.1113, +0.0176] | not shown | 96 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | 3-5 | -0.0790 | [-0.1163, -0.0417] | hurts | 288 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | 6-10 | -0.0211 | [-0.0440, +0.0018] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | 11-15 | -0.0090 | [-0.0255, +0.0075] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_direct | improved / unchanged / degraded | 347 / 533 / 464 | identical scores 131 of 1344 | | |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | all | +0.0136 | [-0.0004, +0.0276] | not shown | 1344 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | 2 | +0.0729 | [-0.0049, +0.1507] | not shown | 96 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | 3-5 | +0.0406 | [-0.0004, +0.0817] | not shown | 288 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | 6-10 | -0.0029 | [-0.0262, +0.0204] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | 11-15 | +0.0020 | [-0.0101, +0.0140] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32 - agent_v2_p2xxl_uw8x4_fp32 | improved / unchanged / degraded | 378 / 627 / 339 | identical scores 242 of 1344 | | |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | all | -0.0023 | [-0.0104, +0.0058] | not shown | 1344 |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 96 |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | 3-5 | +0.0052 | [-0.0082, +0.0186] | not shown | 288 |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | 6-10 | -0.0082 | [-0.0267, +0.0102] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | 11-15 | -0.0014 | [-0.0119, +0.0090] | not shown | 480 |
| agent_v2_p2xxl_gw8x4_fp32~aa - agent_v2_p2xxl_gw8x4_fp32 | improved / unchanged / degraded | 265 / 812 / 267 | identical scores 613 of 1344 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_p2xxl_gw8x4_fp32 | 1344 | 0.75254 | 0 | 0 | 0 | 0 | 0 | 0 | 142.3 | 17.722 | 19.941 | 26.344 | `2e1afee953ee` |
| agent_v2_p2xxl_direct | 1344 | 0.78356 | 0 | 0 | 0 | 0 | 0 | 0 | 71.8 | 1.074 | 3.801 | 5.271 | `d5329fc7f69e` |
| agent_v2_p2xxl_uw8x4_fp32 | 1344 | 0.73896 | 0 | 0 | 0 | 0 | 0 | 0 | 147.1 | 17.754 | 20.428 | 29.128 | `216b9000796b` |
| agent_v2_p2xxl_gw8x4_fp32~aa | 1344 | 0.7502 | 0 | 0 | 0 | 0 | 0 | 0 | 143.0 | 17.721 | 20.21 | 27.888 | `2e1afee953ee` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
