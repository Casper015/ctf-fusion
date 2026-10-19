# P-only paired screen

Schedule `3f18319e8916af678d8229b8c5a448c3f00703bff480224d19f2ee58bd417829`, seed 20262171, 784 fixtures over team counts 2-15 (56 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.4977 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 4 comparisons (alpha 0.05): their intervals and verdicts use z = 2.4977 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.4977) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| ctf_duel_challenger - meowmind_search_on | all | +0.0112 | [-0.0064, +0.0288] | not shown | 784 |
| ctf_duel_challenger - meowmind_search_on | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| ctf_duel_challenger - meowmind_search_on | 3-5 | +0.0298 | [-0.0223, +0.0819] | not shown | 168 |
| ctf_duel_challenger - meowmind_search_on | 6-10 | +0.0049 | [-0.0246, +0.0343] | not shown | 280 |
| ctf_duel_challenger - meowmind_search_on | 11-15 | +0.0085 | [-0.0156, +0.0327] | not shown | 280 |
| ctf_duel_challenger - meowmind_search_on | improved / unchanged / degraded | 253 / 310 / 221 | identical scores 3 of 784 | | |
| ctf_duel_challenger - agent_v2_p2xxl_direct | all | +0.0668 | [+0.0480, +0.0857] | helps | 784 |
| ctf_duel_challenger - agent_v2_p2xxl_direct | 2 | +0.0268 | [-0.0227, +0.0763] | not shown | 56 |
| ctf_duel_challenger - agent_v2_p2xxl_direct | 3-5 | +0.0727 | [+0.0232, +0.1221] | helps | 168 |
| ctf_duel_challenger - agent_v2_p2xxl_direct | 6-10 | +0.0789 | [+0.0466, +0.1112] | helps | 280 |
| ctf_duel_challenger - agent_v2_p2xxl_direct | 11-15 | +0.0593 | [+0.0318, +0.0868] | helps | 280 |
| ctf_duel_challenger - agent_v2_p2xxl_direct | improved / unchanged / degraded | 349 / 277 / 158 | identical scores 0 of 784 | | |
| meowmind_p3_xxl - meowmind_search_on | all | +0.0336 | [+0.0175, +0.0497] | helps | 784 |
| meowmind_p3_xxl - meowmind_search_on | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| meowmind_p3_xxl - meowmind_search_on | 3-5 | +0.0561 | [+0.0113, +0.1008] | helps | 168 |
| meowmind_p3_xxl - meowmind_search_on | 6-10 | +0.0335 | [+0.0063, +0.0607] | helps | 280 |
| meowmind_p3_xxl - meowmind_search_on | 11-15 | +0.0271 | [+0.0032, +0.0510] | helps | 280 |
| meowmind_p3_xxl - meowmind_search_on | improved / unchanged / degraded | 288 / 309 / 187 | identical scores 3 of 784 | | |
| meowmind_p3_distill_xxl - meowmind_search_on | all | +0.0336 | [+0.0170, +0.0502] | helps | 784 |
| meowmind_p3_distill_xxl - meowmind_search_on | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| meowmind_p3_distill_xxl - meowmind_search_on | 3-5 | +0.0536 | [+0.0058, +0.1014] | helps | 168 |
| meowmind_p3_distill_xxl - meowmind_search_on | 6-10 | +0.0317 | [+0.0032, +0.0602] | helps | 280 |
| meowmind_p3_distill_xxl - meowmind_search_on | 11-15 | +0.0302 | [+0.0072, +0.0532] | helps | 280 |
| meowmind_p3_distill_xxl - meowmind_search_on | improved / unchanged / degraded | 280 / 327 / 177 | identical scores 1 of 784 | | |
| ctf_duel_challenger~aa - ctf_duel_challenger | all | -0.0043 | [-0.0139, +0.0054] | not shown | 784 |
| ctf_duel_challenger~aa - ctf_duel_challenger | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| ctf_duel_challenger~aa - ctf_duel_challenger | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 168 |
| ctf_duel_challenger~aa - ctf_duel_challenger | 6-10 | -0.0109 | [-0.0289, +0.0072] | not shown | 280 |
| ctf_duel_challenger~aa - ctf_duel_challenger | 11-15 | -0.0010 | [-0.0213, +0.0192] | not shown | 280 |
| ctf_duel_challenger~aa - ctf_duel_challenger | improved / unchanged / degraded | 183 / 406 / 195 | identical scores 242 of 784 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ctf_duel_challenger | 784 | 0.84285 | 0 | 0 | 0 | 0 | 0 | 0 | 80.6 | 10.722 | 13.199 | 14.465 | `5cf33419427f` |
| meowmind_search_on | 784 | 0.83169 | 0 | 0 | 0 | 0 | 0 | 0 | 77.3 | 10.694 | 13.047 | 14.025 | `ece986d0cbd7` |
| meowmind_p3_xxl | 784 | 0.86533 | 0 | 0 | 0 | 0 | 0 | 0 | 76.4 | 10.566 | 12.887 | 13.86 | `93ce877ed014` |
| meowmind_p3_distill_xxl | 784 | 0.86526 | 0 | 0 | 0 | 0 | 0 | 0 | 78.1 | 10.56 | 12.907 | 14.53 | `bcffab2270ca` |
| agent_v2_p2xxl_direct | 784 | 0.77602 | 0 | 0 | 0 | 0 | 0 | 0 | 46.6 | 0.858 | 2.535 | 4.152 | `d5329fc7f69e` |
| ctf_duel_challenger~aa | 784 | 0.8386 | 0 | 0 | 0 | 0 | 0 | 0 | 83.3 | 10.742 | 13.653 | 15.32 | `5cf33419427f` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
