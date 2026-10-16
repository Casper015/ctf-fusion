# P-only paired screen

Schedule `b54ad55e2673bd3155dfefec898b28bd4ac146fa69243cf98a445bc769d829fa`, seed 20261958, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | Interval (95%) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | all | +0.0164 | [-0.0042, +0.0369] | not shown | 392 |
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | 2 | +0.0179 | [-0.0171, +0.0529] | not shown | 28 |
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | 3-5 | +0.0645 | [+0.0084, +0.1206] | helps | 84 |
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | 6-10 | -0.0037 | [-0.0358, +0.0284] | not shown | 140 |
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | 11-15 | +0.0073 | [-0.0260, +0.0405] | not shown | 140 |
| agent_v2_p2xxl_direct - meow_rl_aligned_p2_large | improved / unchanged / degraded | 140 / 125 / 127 | identical scores 1 of 392 | | |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| agent_v2_p2xxl_direct~aa - agent_v2_p2xxl_direct | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_p2xxl_direct | 392 | 0.7776 | 0 | 0 | 0 | 0 | 0 | 0 | 84.4 | 0.97 | 5.032 | 7.557 | `0c6b1f2e86d1` |
| meow_rl_aligned_p2_large | 392 | 0.76124 | 0 | 0 | 0 | 0 | 0 | 0 | 87.8 | 0.943 | 4.236 | 10.059 | `d76921a19675` |
| agent_v2_p2xxl_direct~aa | 392 | 0.7776 | 0 | 0 | 0 | 0 | 0 | 0 | 103.0 | 0.976 | 5.366 | 13.321 | `0c6b1f2e86d1` |

## Meowmind RL counters (our seat, the last cumulative MEOWRL line per game)

| Arm | Games | With counters | Turns counted / played | Refused | Search used | Search fallback | Leaves |
|---|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large | 392 | 392 | 156800 / 156800 | 0 | 0 | 0 | 0 |

## Network calls per turn and search samples (our seat)

| Arm | Turns | Views | Team encodes | Act heads | Target heads | Fallbacks | Inconsistent | Flag-return states | Expectimax calls | Samples per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| meow_rl_aligned_p2_large |  |  |  |  |  |  |  |  |  |  |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
