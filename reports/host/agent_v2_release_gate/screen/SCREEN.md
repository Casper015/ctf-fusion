# P-only paired screen

Schedule `f510d2a8476048a4b9db8fe6cc8bd690a8953a5b015c95042bc241b0c9b84e54`, seed 20262141, 1344 fixtures over team counts 2-15 (96 maps per count). Utility: share of the other teams beaten (ties half). Intervals: 95%, map groups per count, counts weighted equally. improved/unchanged/degraded count fixtures by the sign of the paired difference; they are not wins.

## Paired differences

| Comparison | Band | Mean | Interval (95%) | Verdict | Fixtures |
|---|---|---:|---|---|---:|
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | all | +0.0011 | [-0.0116, +0.0138] | not shown | 1344 |
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | 2 | +0.0104 | [-0.0279, +0.0488] | not shown | 96 |
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | 3-5 | +0.0130 | [-0.0237, +0.0498] | not shown | 288 |
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | 6-10 | -0.0119 | [-0.0330, +0.0092] | not shown | 480 |
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | 11-15 | +0.0052 | [-0.0115, +0.0219] | not shown | 480 |
| agent_v2_rl_general_csp - agent_v2_p2xxl_direct | improved / unchanged / degraded | 457 / 427 / 460 | identical scores 0 of 1344 | | |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | all | +0.0000 | [+0.0000, +0.0000] | not shown | 1344 |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | 2 | +0.0000 | [+0.0000, +0.0000] | not shown | 96 |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | 3-5 | +0.0000 | [+0.0000, +0.0000] | not shown | 288 |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | 6-10 | +0.0000 | [+0.0000, +0.0000] | not shown | 480 |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | 11-15 | +0.0000 | [+0.0000, +0.0000] | not shown | 480 |
| agent_v2_rl_general_csp~aa - agent_v2_rl_general_csp | improved / unchanged / degraded | 0 / 1344 / 0 | identical scores 1344 of 1344 | | |

## Arms

| Arm | Games | Utility | Faults | Dead | Late | Opponent faults | Opponent late | Opponent dead | First reply max ms | p99 median ms | p99 worst game ms | Max reply ms | Binary |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| agent_v2_rl_general_csp | 1344 | 0.76694 | 0 | 0 | 0 | 0 | 0 | 0 | 102.0 | 1.346 | 5.656 | 15.203 | `08d1d5602e4f` |
| agent_v2_p2xxl_direct | 1344 | 0.7658 | 0 | 0 | 0 | 0 | 0 | 0 | 94.8 | 1.514 | 6.905 | 13.403 | `d5329fc7f69e` |
| agent_v2_rl_general_csp~aa | 1344 | 0.76694 | 0 | 0 | 0 | 0 | 0 | 0 | 117.2 | 1.384 | 5.95 | 15.151 | `08d1d5602e4f` |

Max reply is the largest single reply over every game; p99 is per game (not pooled). A late reply counts as a fault. Re-run noise rows (X~aa - X) replay an identical binary.
