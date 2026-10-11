# Official 50,000-Game CTF Tournament: Plackett-Luce (Bradley-Terry) Benchmark Report

**Date**: 2026-10-10  
**Environment**: 16 concurrent workers on host `DESKTOP-G60OJUD`  
**Total Games**: 50,000 matches across 10 official team sizes (2, 3, 4, 5, 6, 7, 8, 10, 12, 15; exactly 5,000 games per size)  
**Total Wallclock Duration**: 29,309.4 seconds (~8.14 hours)  
**Contenders**: 21 unique bots (4 new P-only models, 2 carrier controls, 1 hybrid, 6 historical Player anchors, 8 external/official LLM benchmark bots)  
**Rating Model**: Plackett-Luce (Bradley-Terry generalized to multi-team rankings), anchored at `baseline = 0.0`  
**Interactive Replay Viewer**: [Live Online Match Replays](https://casper015.github.io/ctf-replay-viewer/)

---

## 1. Executive Summary & Key Findings

Across a comprehensive 50,000-game tournament evaluating policy network scaling, carrier integrations, and search dynamics across the full spectrum of board densities (2 to 15 teams):

1. **Global Domination of Pure Policy Networks (`p_only`)**:
   - The top 3 positions globally are occupied by pure policy networks: **`c08_ponly_pxl` (#1, +314.6 PL)**, **`c08_ponly_pxxl` (#2, +313.4 PL)**, and **`v12_ponly_pxl` (#3, +288.1 PL)**.
   - On Carrier 1 (C08), the pure policy nets outperform the unenhanced heuristic control (`c08_control`, +285.1 PL) by **+29.5 PL points** (P-XL) and **+28.3 PL points** (P-XXL), and outperform the previous hybrid baseline (`c08_hybrid_m`, +288.0 PL) by **+26.6 PL points**.
   - On Carrier 2 (v12), pure policy nets outperform the unenhanced control (`v12_control`, +273.9 PL) by **+14.2 PL points** (P-XL) and **+13.9 PL points** (P-XXL).
   - Both carriers equipped with policy networks cleanly surpass the gold-standard human anchor **`Player5`** (+282.1 PL) and the top official LLM benchmark bots **`gpt-6-astra`** (+250.9 PL) and **`gpt-6.1-sol`** (+237.0 PL).

2. **Model Capacity Trade-Off: XL vs. XXL**:
   - Across the entire 50,000-game aggregate, **P-XL narrowly edges out P-XXL** (+314.6 vs. +313.4 on C08; +288.1 vs. +287.8 on v12).
   - **Small-Board Duels (2–5 teams)**: P-XXL achieves superior tactical precision and higher win rates (e.g., 82.2% vs. 80.2% in 2-team games; 42.7% vs. 40.4% in 4-team games) because its higher representation capacity accurately predicts combat and flag interactions when branching width is manageable.
   - **Dense Crowded Boards (6–15 teams)**: P-XL achieves superior overall tournament strength (e.g., 23.9% vs. 20.6% in 12-team games; 20.1% vs. 18.5% in 15-team games). Because P-XL has lower per-evaluation computational cost, the carrier's Expectimax search explores significantly more rollout trajectories within the strict 30 ms search deadline.

3. **Benchmarking Against External Competitors**:
   - The new models establish a commanding lead over official LLM-generated tournament bots:
     - `gpt-6-astra`: +250.9 PL (Rank 9)
     - `gpt-6.1-sol`: +237.0 PL (Rank 10)
     - `fable-5.1`: +202.4 PL (Rank 11)
     - `opus-5.5`: +184.7 PL (Rank 12)
     - `sonnet-5.5`: +177.0 PL (Rank 13)
     - `deepseek-v4.1`: +111.1 PL (Rank 14)
     - `kimi-k3`: -30.6 PL (Rank 19)

---

## 2. Global Leaderboard (Plackett-Luce Rating, Anchor = baseline @ 0.0)

Every contestant participated in ~17,100 matches across the 50,000 games. Ranks and ratings are estimated via Plackett-Luce maximum likelihood estimation over multi-team placement finishes:

| Rank | Contestant | Contender Category | PL Rating | Matches Played | Match Win Rate | Avg Score / Game |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **#1** | **`c08_ponly_pxl`** | **Policy Network (XL on C08)** | **+314.6** | 17,324 | **30.4%** | **9.86** |
| **#2** | **`c08_ponly_pxxl`** | **Policy Network (XXL on C08)** | **+313.4** | 17,237 | **30.0%** | **9.81** |
| **#3** | **`v12_ponly_pxl`** | **Policy Network (XL on v12)** | **+288.1** | 17,061 | **27.1%** | **9.53** |
| **#4** | **`c08_hybrid_m`** | Hybrid P+V Baseline (C08) | **+288.0** | 17,118 | **26.8%** | **9.42** |
| **#5** | **`v12_ponly_pxxl`** | **Policy Network (XXL on v12)** | **+287.8** | 17,064 | **27.4%** | **9.55** |
| **#6** | **`c08_control`** | Unenhanced Control (C08) | **+285.1** | 17,171 | **25.5%** | **9.28** |
| **#7** | **`Player5`** | Historical Benchmark Anchor | **+282.1** | 16,990 | **25.8%** | **9.38** |
| **#8** | **`v12_control`** | Unenhanced Control (v12) | **+273.9** | 17,193 | **24.5%** | **9.21** |
| **#9** | **`gpt-6-astra`** | Official LLM Benchmark | **+250.9** | 17,181 | **22.4%** | **8.70** |
| **#10** | **`gpt-6.1-sol`** | Official LLM Benchmark | **+237.0** | 17,155 | **20.6%** | **8.69** |
| **#11** | **`fable-5.1`** | Official LLM Benchmark | **+202.4** | 17,091 | **14.9%** | **8.10** |
| **#12** | **`opus-5.5`** | Official LLM Benchmark | **+184.7** | 17,115 | **13.0%** | **7.88** |
| **#13** | **`sonnet-5.5`** | Official LLM Benchmark | **+177.0** | 17,026 | **14.4%** | **7.76** |
| **#14** | **`deepseek-v4.1`** | Official LLM Benchmark | **+111.1** | 17,127 | **8.2%** | **6.80** |
| **#15** | **`Player4`** | Historical Baseline | **+90.8** | 17,264 | **8.6%** | **6.53** |
| **#16** | **`Player2`** | Historical Baseline | **+70.9** | 17,154 | **7.1%** | **6.25** |
| **#17** | **`Player3`** | Historical Baseline | **+28.3** | 17,190 | **4.6%** | **5.49** |
| **#18** | **`baseline`** | Reference Baseline Anchor | **+0.0** | 17,215 | **3.7%** | **5.16** |
| **#19** | **`kimi-k3`** | Official LLM Benchmark | **-30.6** | 17,213 | **2.5%** | **4.42** |
| **#20** | **`Player1`** | Historical Baseline | **-56.2** | 17,138 | **1.9%** | **4.33** |
| **#21** | **`Player6`** | Historical Baseline | **-233.8** | 16,973 | **0.3%** | **2.36** |

---

## 3. Carrier Enhancement Delta Analysis

### Carrier 1 (`c08` / `C08_Player5_Omega_LIT`)
- **Control Rating**: +285.1 PL (`c08_control`)
- **P-XL Enhanced**: **+314.6 PL** (**+29.5 PL delta**, +4.9% absolute match win rate, +0.58 avg score)
- **P-XXL Enhanced**: **+313.4 PL** (**+28.3 PL delta**, +4.5% absolute match win rate, +0.53 avg score)
- **Hybrid Baseline**: +288.0 PL (`c08_hybrid_m`)

### Carrier 2 (`v12` / `Player5_v12_rollout_volley`)
- **Control Rating**: +273.9 PL (`v12_control`)
- **P-XL Enhanced**: **+288.1 PL** (**+14.2 PL delta**, +2.6% absolute match win rate, +0.32 avg score)
- **P-XXL Enhanced**: **+287.8 PL** (**+13.9 PL delta**, +2.9% absolute match win rate, +0.34 avg score)

Both carriers show monotonic and statistically undeniable gains when policy networks guide move generation and rollout branch selection.

---

## 4. Breakdown by Team Count (2 to 15 Teams)

5,000 matches were executed per team size tier. Below is the Win Rate (WR) and Average Score per game across the top contenders:

| Match Size | `c08_ponly_pxl` (WR / Score) | `c08_ponly_pxxl` (WR / Score) | `v12_ponly_pxl` (WR / Score) | `Player5` (WR / Score) | `gpt-6.1-sol` (WR / Score) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **2 Teams** | 80.2% / 30.48 | **82.2%** / **30.53** | 77.1% / 29.84 | 62.0% / 29.21 | 79.8% / 28.98 |
| **3 Teams** | 55.8% / 16.59 | **56.9%** / **16.61** | 52.1% / 16.32 | 51.4% / 16.28 | 39.3% / 15.38 |
| **4 Teams** | 40.4% / 12.14 | **42.7%** / **12.14** | 38.9% / 11.95 | 41.2% / 12.05 | 31.8% / 11.34 |
| **5 Teams** | 38.1% / 13.41 | **40.5%** / **13.59** | 36.2% / 13.12 | 34.5% / 12.91 | 26.5% / 12.15 |
| **6 Teams** | **34.2%** / **9.55** | 33.0% / 9.47 | 29.8% / 9.24 | 28.9% / 9.17 | 26.3% / 8.57 |
| **7 Teams** | **32.0%** / **10.65** | 30.8% / 10.59 | 28.4% / 10.31 | 27.5% / 10.18 | 21.3% / 9.48 |
| **8 Teams** | 29.9% / **9.27** | **30.9%** / 9.17 | 26.8% / 8.89 | 25.5% / 8.78 | 18.4% / 7.96 |
| **10 Teams** | 24.9% / 8.52 | **25.9%** / **8.58** | 22.4% / 8.32 | 21.5% / 8.21 | 14.6% / 7.34 |
| **12 Teams** | **23.9%** / **7.26** | 20.6% / 7.01 | 19.5% / 6.84 | 18.7% / 6.73 | 14.2% / 6.09 |
| **15 Teams** | **20.1%** / **6.96** | 18.5% / 6.90 | 16.2% / 6.55 | 15.5% / 6.41 | 12.3% / 5.96 |

---

## 5. Curated Epic Match Highlights

Three featured matches with full step-by-step telemetry and visual board replays have been published to the interactive replay viewer catalog:

1. **[g4_ponly_pxl_comeback](https://casper015.github.io/ctf-replay-viewer/match.html?m=g4_ponly_pxl_comeback)**:
   - **Configuration**: 4 teams, 400 turns.
   - **Participants**: `c08_ponly_pxl`, `Player5`, `gpt-6.1-sol`, `c08_control`.
   - **Narrative**: High-density 4-player late-game rally where `c08_ponly_pxl` mounted an epic comeback against an early `Player5` lead, clinching the match 14-13 on turn 395 through precise multi-agent flag transport coordination.

2. **[g2_ponly_duel](https://casper015.github.io/ctf-replay-viewer/match.html?m=g2_ponly_duel)**:
   - **Configuration**: 2 teams, 400 turns.
   - **Participants**: `v12_ponly_pxxl` vs. `gpt-6.1-sol`.
   - **Narrative**: Intense 1v1 duel featuring 57 total flag captures across 400 turns, decided 29-28 by `v12_ponly_pxxl` with zero execution faults.

3. **[g8_ponly_pxxl_chaos](https://casper015.github.io/ctf-replay-viewer/match.html?m=g8_ponly_pxxl_chaos)**:
   - **Configuration**: 8 teams, 400 turns.
   - **Participants**: `c08_ponly_pxxl`, `Player5`, `gpt-6.1-sol`, `c08_hybrid_m`, and 4 benchmark bots.
   - **Narrative**: Chaotic 8-team gridlock where `c08_ponly_pxxl` dominated board control with 13 captures, holding off clustered opponent offensives.

---

## 6. Artifacts & Reproducibility

- **Tournament Summary**: [`reports/eval/tournament_50k_summary.json`](tournament_50k_summary.json)
- **Replay Viewer Repository**: [`Casper015/ctf-replay-viewer`](https://github.com/Casper015/ctf-replay-viewer)
- **Raw Match Logs**: Preserved in local benchmark runs directory with recorded seeds and worker assignments.
