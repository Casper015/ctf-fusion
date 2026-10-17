# P-only paired screen

Schedule `7e7faae90fd2c14ba516686b06ba47966323b674b472cd305621230edc7a4751`, seed 20262111, 784 fixtures over team counts 2-15 (56 maps per count). Utility: share of the other teams beaten (ties half). Intervals: z = 2.2414 for the comparisons, 95% for re-run noise, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

**Comparison intervals:** bonferroni over the 2 comparisons (alpha 0.05): their intervals and verdicts use z = 2.2414 (the 95% interval and verdict are in screen.json as ci95 and verdict_95). Re-run noise rows use 95%.

## Paired differences

| Comparison | Band | Mean | Interval (z = 2.2414) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| meowmind_search_on - meowmind_search_off | all | +0.0809 | [+0.0631, +0.0988] | helps | 784 |
| meowmind_search_on - meowmind_search_off | 2 | +0.0536 | [-0.0082, +0.1153] | not shown | 56 |
| meowmind_search_on - meowmind_search_off | 3-5 | +0.0774 | [+0.0334, +0.1214] | helps | 168 |
| meowmind_search_on - meowmind_search_off | 6-10 | +0.0779 | [+0.0468, +0.1090] | helps | 280 |
| meowmind_search_on - meowmind_search_off | 11-15 | +0.0916 | [+0.0657, +0.1175] | helps | 280 |
| meowmind_search_on - meowmind_search_off | improved / unchanged / degraded | 372 / 250 / 162 | identical scores 3 of 784 | | |
| meowmind_search_on - agent_v2_p2xxl_direct | all | +0.0661 | [+0.0488, +0.0834] | helps | 784 |
| meowmind_search_on - agent_v2_p2xxl_direct | 2 | +0.0536 | [-0.0145, +0.1216] | not shown | 56 |
| meowmind_search_on - agent_v2_p2xxl_direct | 3-5 | +0.0494 | [+0.0038, +0.0949] | helps | 168 |
| meowmind_search_on - agent_v2_p2xxl_direct | 6-10 | +0.0724 | [+0.0447, +0.1001] | helps | 280 |
| meowmind_search_on - agent_v2_p2xxl_direct | 11-15 | +0.0725 | [+0.0471, +0.0978] | helps | 280 |
| meowmind_search_on - agent_v2_p2xxl_direct | improved / unchanged / degraded | 350 / 273 / 161 | identical scores 0 of 784 | | |
| meowmind_search_on~aa - meowmind_search_on | all | -0.0007 | [-0.0115, +0.0100] | not shown | 784 |
| meowmind_search_on~aa - meowmind_search_on | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 56 |
| meowmind_search_on~aa - meowmind_search_on | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 168 |
| meowmind_search_on~aa - meowmind_search_on | 6-10 | -0.0019 | [-0.0247, +0.0210] | not shown | 280 |
| meowmind_search_on~aa - meowmind_search_on | 11-15 | -0.0001 | [-0.0197, +0.0194] | not shown | 280 |
| meowmind_search_on~aa - meowmind_search_on | improved / unchanged / degraded | 200 / 381 / 203 | identical scores 225 of 784 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| meowmind_search_on | 784 | 0.83507 | 0 | 0 | 0 | 0 | 0 | 0 | 82.6 | 10.865 | 14.012 | 17.212 | `ece986d0cbd7` |
| meowmind_search_off | 784 | 0.75413 | 0 | 0 | 0 | 0 | 0 | 0 | 53.2 | 1.144 | 3.318 | 4.689 | `047f6a861466` |
| agent_v2_p2xxl_direct | 784 | 0.76894 | 0 | 0 | 0 | 0 | 0 | 0 | 54.4 | 0.946 | 3.055 | 6.297 | `d5329fc7f69e` |
| meowmind_search_on~aa | 784 | 0.83435 | 0 | 0 | 0 | 0 | 0 | 0 | 85.2 | 10.873 | 13.627 | 16.1 | `ece986d0cbd7` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
