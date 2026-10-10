# Official Standard 1,000-Game Tournament: Bradley-Terry (BT) Benchmark Report
**Date**: 2026-10-10  
**Objective**: Evaluate the exact playing strength and Bradley-Terry (BT) score increase of the two Tier L Policy + Value (`pvL`) models against all official benchmark bots and their unenhanced controls across 1,000 games on the 8 official benchmark team sizes (2, 3, 4, 5, 6, 8, 10, 15).

## 1. Executive Summary & Key Findings
Across a full 1,000-game round-robin tournament comprising **16 contenders** (2 pvL models, 2 controls, and 12 official reference benchmarks), the two `pvL` enhanced models achieved the **top two positions on the global leaderboard**:

1. **Carrier 1 (`C08_Player5_Omega_LIT` + pvL)**:
   - **Global Rank #1** at **+36.3 BT** (95% CI: `[-4.1, +80.1]`).
   - **+28.9 BT increase** over its control (`C08_ctrl`: +7.4 BT).
   - Pair Win rate increased from **68.02% to 70.63%** (+2.61%). Match 1st-place win rate reached **32.44%**.
2. **Carrier 2 (`Player5_v12_rollout_volley` + pvL)**:
   - **Global Rank #2** at **+13.3 BT** (95% CI: `[-26.0, +56.8]`).
   - **+21.7 BT increase** over its control (`v12_ctrl`: -8.4 BT).
   - Pair Win rate surged from **64.88% to 69.35%** (+4.47%). Match 1st-place win rate reached **32.15%**.
   - Overcame both `Player5` (Anchor = 0.0) and `Astra` (+10.4 BT) to take Rank #2.
3. **Zero Faults & 100% Timing Compliance**:
   - `v12_pvL` maintained a blazing fast p99 latency of **3.3ms**.
   - `C08_pvL` maintained a p99 latency of **26.1ms** (comfortably under the 50ms engine limit).
   - **0 timeout violations and 0 strikes** across all 1,000 matches.

---
## 2. Official Bradley-Terry (BT) Leaderboard (Anchor = Player5 @ 0.0)
| Rank | Contender | Type | BT Rating | 95% Bootstrap CI | Pair Win % | Match Win Rate | Turn p99 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **`C08_pvL`** | **Enhanced pvL (C08)** | ** +36.3** | [ -4.1, +80.1] | 70.63% | 32.44% | 26.1ms |
| **#2** | **`v12_pvL`** | **Enhanced pvL (v12)** | ** +13.3** | [-26.0, +56.8] | 69.35% | 32.15% | 3.3ms |
| **#3** | **`Astra`** | Official Benchmark | ** +10.4** | [-23.0, +52.8] | 66.44% | 26.65% | 12.6ms |
| **#4** | **`C08_ctrl`** | Control Baseline (C08) | **  +7.4** | [-28.1, +47.5] | 68.02% | 28.17% | 25.7ms |
| **#5** | **`Player5`** | Official Anchor (0.0) | **  +0.0** | [ +0.0,  +0.0] | 68.06% | 31.64% | 1.6ms |
| **#6** | **`v12_ctrl`** | Control Baseline (v12) | **  -8.4** | [-52.0, +33.1] | 64.88% | 27.80% | 2.9ms |
| **#7** | **`Sol`** | Official Benchmark | ** -59.7** | [-103.4, -18.3] | 59.01% | 19.80% | 5.0ms |
| **#8** | **`Opus`** | Official Benchmark | ** -96.2** | [-137.5, -55.5] | 53.39% | 19.03% | 1.3ms |
| **#9** | **`Sonnet`** | Official Benchmark | **-100.8** | [-132.9, -55.7] | 51.82% | 15.78% | 1.2ms |
| **#10** | **`Fable`** | Official Benchmark | **-101.9** | [-141.1, -54.4] | 54.35% | 13.59% | 1.5ms |
| **#11** | **`DeepSeek`** | Official Benchmark | **-131.1** | [-167.8, -81.9] | 46.97% | 13.10% | 1.8ms |
| **#12** | **`Baseline`** | Official Benchmark | **-253.9** | [-289.5, -214.1] | 29.04% | 5.23% | 1.2ms |
| **#13** | **`MiMo`** | Official Benchmark | **-273.4** | [-310.7, -230.2] | 26.64% | 4.95% | 2.0ms |
| **#14** | **`Kimi`** | Official Benchmark | **-286.7** | [-330.8, -243.1] | 24.46% | 3.80% | 1.6ms |
| **#15** | **`Qwen`** | Official Benchmark | **-293.2** | [-336.8, -252.5] | 27.01% | 3.44% | 1.5ms |
| **#16** | **`GLM`** | Official Benchmark | **-349.1** | [-385.4, -313.7] | 21.87% | 1.18% | 1.4ms |

---
## 3. pvL Enhancement Delta Analysis
| Carrier | Control BT | pvL Enhanced BT | **Delta BT** | Control Pair Win | pvL Pair Win | **Delta Pair Win** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Carrier 1 (`C08_Player5_Omega_LIT`)** |   +7.4 |  +36.3 | ** +28.9 pts** | 68.02% | 70.63% | **+2.61%** |
| **Carrier 2 (`Player5_v12_rollout_volley`)** |   -8.4 |  +13.3 | ** +21.7 pts** | 64.88% | 69.35% | **+4.47%** |

---
## 4. Per-Size Bradley-Terry Breakdown
BT ratings fitted individually for each team count size:
| Contender | 2T | 3T | 4T | 5T | 6T | 8T | 10T | 15T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `C08_pvL` | +15.3 | +12.9 | +67.6 | +39.1 | +10.6 |  +1.2 |  -2.4 |  -8.6 |
| `v12_pvL` | +29.2 |  -1.5 | +43.1 |  +9.0 | -12.3 |  +3.9 |  +5.4 |  +5.7 |
| `Astra` | -20.8 | +15.5 | +88.0 |  +7.8 |  -7.2 | -23.9 | -21.9 | -20.6 |
| `C08_ctrl` | -40.6 | +26.4 | +59.5 |  -2.4 |  -4.2 |  -3.3 | -16.2 |  +4.9 |
| `Player5` |  +0.0 |  +0.0 |  +0.0 |  +0.0 |  +0.0 |  +0.0 |  +0.0 |  +0.0 |
| `v12_ctrl` |  +7.8 | +12.5 | +23.3 | -24.3 | -20.6 | -17.1 | -16.8 | -11.9 |
| `Sol` | -13.6 | -87.7 | +20.4 | -30.9 | -45.0 | -57.6 | -25.9 | -19.0 |
| `Opus` | -164.5 | -58.3 | +27.3 |  -1.7 | -55.9 | -72.3 | -53.9 | -58.3 |
| `Sonnet` | -133.5 | -65.4 | +50.8 | -32.9 | -76.7 | -72.5 | -56.8 | -62.5 |
| `Fable` | -193.6 | -53.7 | -18.2 | -47.2 | -27.1 | -53.5 | -37.0 | -56.8 |
| `DeepSeek` | -115.8 | -38.0 | -16.1 | -41.4 | -112.5 | -92.6 | -92.5 | -64.2 |
| `Baseline` | -238.8 | -131.0 | -63.7 | -145.2 | -90.0 | -136.1 | -144.2 | -153.4 |
| `MiMo` | -296.3 | -85.1 | -89.9 | -128.1 | -124.8 | -172.4 | -160.4 | -149.4 |
| `Kimi` | -257.6 | -93.2 | -55.1 | -170.4 | -177.3 | -174.3 | -162.5 | -154.4 |
| `Qwen` | -341.9 | -110.9 | -97.0 | -136.0 | -138.0 | -144.2 | -161.6 | -135.7 |
| `GLM` | -401.7 | -146.0 | -136.7 | -162.2 | -193.2 | -184.7 | -161.6 | -134.8 |

---
## 5. Artifacts and Raw Data
- **Summary JSON**: [`reports/eval/tournament_1000_bt.json`](file:///reports/tournament_1000_bt.json)
- **Match-by-Match Logs (1,000 matches)**: [`reports/eval/tournament_1000_matches.jsonl`](file:///reports/tournament_1000_matches.jsonl)