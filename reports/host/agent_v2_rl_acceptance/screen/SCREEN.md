# P-only paired screen

Schedule `bbab7d975e5c2af6f504b79684afc200ce58d529d4342159aec9fa6f3584aba1`, seed 20262101, 1344 fixtures over team counts 2-15 (96 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.2414 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 2 comparisons (alpha 0.05): their intervals and verdicts use z = 2.2414 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.2414) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_rl_csp - agent_v2_rl_general_csp | all | -0.0277 | [-0.0406, -0.0148] | hurts | 1344 |
| agent_v2_rl_csp - agent_v2_rl_general_csp | 2 | -0.0104 | [-0.0391, +0.0182] | not shown | 96 |
| agent_v2_rl_csp - agent_v2_rl_general_csp | 3-5 | -0.0279 | [-0.0630, +0.0072] | not shown | 288 |
| agent_v2_rl_csp - agent_v2_rl_general_csp | 6-10 | -0.0316 | [-0.0534, -0.0098] | hurts | 480 |
| agent_v2_rl_csp - agent_v2_rl_general_csp | 11-15 | -0.0271 | [-0.0459, -0.0083] | hurts | 480 |
| agent_v2_rl_csp - agent_v2_rl_general_csp | improved / unchanged / degraded | 389 / 431 / 524 | identical scores 1 of 1344 | | |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | all | +0.0290 | [+0.0157, +0.0422] | helps | 1344 |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | 2 | -0.0156 | [-0.0416, +0.0104] | not shown | 96 |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | 3-5 | +0.0273 | [-0.0073, +0.0620] | not shown | 288 |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | 6-10 | +0.0213 | [-0.0011, +0.0436] | not shown | 480 |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | 11-15 | +0.0465 | [+0.0263, +0.0668] | helps | 480 |
| agent_v2_rl_general_csp - agent_v2_rl_init_csp | improved / unchanged / degraded | 520 / 422 / 402 | identical scores 1 of 1344 | | |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | all | +0.0000 | [+0.0000, +0.0000] | not shown | 1344 |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 96 |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 288 |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 480 |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 480 |
| agent_v2_rl_csp~aa - agent_v2_rl_csp | improved / unchanged / degraded | 0 / 1344 / 0 | identical scores 1344 of 1344 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_rl_csp | 1344 | 0.76079 | 0 | 0 | 0 | 0 | 0 | 0 | 71.1 | 1.119 | 5.043 | 12.523 | `458f51f068b2` |
| agent_v2_rl_general_csp | 1344 | 0.78849 | 0 | 0 | 0 | 0 | 0 | 0 | 61.3 | 1.139 | 4.237 | 6.572 | `08d1d5602e4f` |
| agent_v2_rl_init_csp | 1344 | 0.75954 | 0 | 0 | 0 | 0 | 0 | 0 | 64.5 | 1.141 | 4.67 | 16.405 | `7e4280feda0d` |
| agent_v2_rl_csp~aa | 1344 | 0.76079 | 0 | 0 | 0 | 0 | 0 | 0 | 63.0 | 1.12 | 4.373 | 12.705 | `458f51f068b2` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
