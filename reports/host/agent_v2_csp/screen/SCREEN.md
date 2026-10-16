# P-only paired screen

Schedule `b30654ae120f289ddc180c82203f436a09bab536000f80713eef3d48708b48fd`, seed 20261961, 392 fixtures over team counts 2-15 (28 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | Interval (95%) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_csp_direct - agent_v2_p2xxl_direct | all | -0.0116 | [-0.0315, +0.0082] | not shown | 392 |
| agent_v2_csp_direct - agent_v2_p2xxl_direct | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| agent_v2_csp_direct - agent_v2_p2xxl_direct | 3-5 | -0.0481 | [-0.1060, +0.0097] | not shown | 84 |
| agent_v2_csp_direct - agent_v2_p2xxl_direct | 6-10 | -0.0090 | [-0.0406, +0.0227] | not shown | 140 |
| agent_v2_csp_direct - agent_v2_p2xxl_direct | 11-15 | +0.0053 | [-0.0244, +0.0350] | not shown | 140 |
| agent_v2_csp_direct - agent_v2_p2xxl_direct | improved / unchanged / degraded | 120 / 138 / 134 | identical scores 5 of 392 | | |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | all | +0.0000 | [+0.0000, +0.0000] | not shown | 392 |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 28 |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 84 |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 140 |
| agent_v2_csp_direct~aa - agent_v2_csp_direct | improved / unchanged / degraded | 0 / 392 / 0 | identical scores 392 of 392 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_csp_direct | 392 | 0.76448 | 0 | 0 | 0 | 0 | 0 | 0 | 51.3 | 1.085 | 3.429 | 5.653 | `83b80ff99b41` |
| agent_v2_p2xxl_direct | 392 | 0.77611 | 0 | 0 | 0 | 0 | 0 | 0 | 54.6 | 1.052 | 3.648 | 6.052 | `119c7f76b972` |
| agent_v2_csp_direct~aa | 392 | 0.76448 | 0 | 0 | 0 | 0 | 0 | 0 | 55.8 | 1.082 | 3.5 | 5.319 | `83b80ff99b41` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
