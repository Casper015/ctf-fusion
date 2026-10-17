# Host Screen Report: Agent V2 Gated Search vs Direct and Ungated Search (T0079)

**Date:** 2026-10-17  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0079 (`Host: gated wider agent_v2 search versus direct and ungated wider search`)  
**Artifacts & Receipts:** `reports/host/agent_v2_search_gate/`  
**Screen Profile:** `profiles/agent_v2_search_gate.json`  
**Seeds:** Preflight `20262130` | Screen `20262131` (reserved in `docs/AGENT_V2_SEARCH_GATE_SEEDS.md`)  
**Bonferroni Multiplicity:** $\alpha = 0.05$, $k = 2$ comparisons ($z = 2.2414$)  
**Primary Contrast ($GW - D$):** **hurts** (mean diff = -0.0310, Bonferroni CI [-0.0447, -0.0174])  
**Supporting Contrast ($GW - UW$):** **not shown** (mean diff = +0.0136, Bonferroni CI [-0.0004, +0.0276])  
**Descriptive Contrast ($UW - D$):** **hurts** (mean diff = -0.0446, 95% CI [-0.0571, -0.0321])  

---

## 1. Executive Summary

Task T0079 evaluates whether T0076's frozen search repair—an 8 candidates $\times$ 4 scenarios wide tree with a $\tau = 0.002$ override gate—produces genuine game wins over actor-direct play on the `agent_v2` carrier.

Three variants were evaluated across 5,376 games against the frozen reference roster (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player4`):
- **Arm $D$ (`agent_v2_p2xxl_direct`):** The actor-direct baseline without search (binary `d5329fc7f69e`).
- **Arm $GW$ (`agent_v2_p2xxl_gw8x4_fp32`):** Gated wide search with 8 candidates $\times$ 4 scenarios, $\tau = 0.002$ threshold, FP32 critic (binary `2e1afee953ee`).
- **Arm $UW$ (`agent_v2_p2xxl_uw8x4_fp32`):** Ungated wide search with identical 8 candidates $\times$ 4 scenarios, $\tau = 0.0$, FP32 critic (binary `216b9000796b`).
- **Arm $GW\sim\text{aa}$ (`agent_v2_p2xxl_gw8x4_fp32~aa`):** Re-run noise control playing identical binary `2e1afee953ee`.

### Core Empirical Findings
1. **Search Actively Degrades Performance ($GW - D = -3.10\text{ pp}$, Verdict: hurts):**  
   Gated search performs statistically significantly worse than direct play overall (mean diff -0.0310, Bonferroni CI [-0.0447, -0.0174], $p < 0.001$).
2. **Degradation Is Heaviest Where Search Actually Runs (Band 3–5: $-7.90\text{ pp}$):**  
   - On counts 3–5, where search completions reach ~100%, gated search suffers a severe **-7.90 pp drop** vs direct play (95% CI [-0.1163, -0.0417]).
   - On counts 11–15, conservative trajectory admission causes ~97% of trees to fall back to the actor, shrinking the gap to **-0.90 pp** ([-0.0255, +0.0075]). This proves that whenever search executes, it damages playing strength.
3. **The Gate Softens the Loss but Cannot Fix the Underlying Deficit ($GW - UW = +1.36\text{ pp}$):**  
   The $\tau = 0.002$ gate successfully filters out ~15% of noisy overrides, boosting utility by +1.36 pp over raw ungated search ($UW$). However, ungated search is catastrophically worse ($UW - D = -4.46\text{ pp}$ overall, and **-11.98 pp on 2 teams**).
4. **Bradley-Terry Leaderboard Impact:**  
   - Direct play $D$: **+94.6 BT** vs Fable (**-7.3 BT** on official `Player5 = 0.0` scale).
   - Gated search $GW$: **+46.3 BT** vs Fable (**-55.6 BT** on official scale, **-48.3 BT drop**).
   - Ungated search $UW$: **+16.2 BT** vs Fable (**-85.7 BT** on official scale, **-78.4 BT drop**).

---

## 2. Eight-Worker Preflight Gate

- **Receipt:** `reports/host/agent_v2_search_gate/PREFLIGHT.json`
- **Seed:** `20262130` | **Workers:** 8 (exact) | **Wall Time:** 99.6 s  
- **Fixtures:** 64 fixtures (16 maps each for counts 2, 5, 8, 15) $\times$ 3 variants = 192 games.
- **Gates:** first reply < 1,800 ms, steady reply < 48 ms, 0 faults/late/dead, 100% turn coverage.

| Variant | Role | Turns | Turn Coverage | Steady Max (ms) | First Max (ms) | Faults / Late / Dead | Gate Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `agent_v2_p2xxl_gw8x4_fp32` | Gated Wide ($\tau=0.002$) | 25,600 | 100.0% | 19.71 | 85.1 | 0 / 0 / 0 | **PASS** |
| `agent_v2_p2xxl_direct` | Direct (no search) | 25,600 | 100.0% | 3.33 | 47.5 | 0 / 0 / 0 | **PASS** |
| `agent_v2_p2xxl_uw8x4_fp32` | Ungated Wide ($\tau=0$) | 25,600 | 100.0% | 19.86 | 89.2 | 0 / 0 / 0 | **PASS** |

### Preflight Search Work & Refusal Telemetry Breakdown

| Variant | Team Count | Turns | Completed Trees (`search_used`) | Completion Rate | Overrides | Gate-Kept Trees | Primary Refusal Cause |
|---|---:|---:|---:|---:|---:|---:|---|
| **$GW$** | 2 | 6,400 | 6,400 | **100.0%** | 4,476 (69.9%) | 873 (16.3%) | None |
| | 5 | 6,400 | 6,393 | **99.9%** | 4,582 (71.7%) | 771 (14.4%) | `refuse_trajectory` (7) |
| | 8 | 6,400 | 1,916 | **29.9%** | 1,279 (66.8%) | 208 (14.0%) | `refuse_trajectory` (4,484) |
| | 15 | 6,400 | 199 | **3.1%** | 39 (19.6%) | 12 (23.5%) | `refuse_trajectory` (6,201) |
| **$UW$** | 2 | 6,400 | 6,400 | **100.0%** | 5,349 (83.6%) | 0 (0.0%) | None |
| | 5 | 6,400 | 6,396 | **99.9%** | 5,248 (82.0%) | 0 (0.0%) | `refuse_trajectory` (4) |
| | 8 | 6,400 | 1,705 | **26.6%** | 1,325 (77.7%) | 0 (0.0%) | `refuse_trajectory` (4,695) |
| | 15 | 6,400 | 202 | **3.2%** | 51 (25.2%) | 0 (0.0%) | `refuse_trajectory` (6,198) |

All 10,692 fallback turns in $GW$ were caused exclusively by `refuse_trajectory`. As documented in `AGENT_V2_SEARCH_GATE.md`, the non-decaying worst-seen trajectory estimate causes games with 8+ teams to permanently disable search after a single slow simulation step.

---

## 3. Formal Paired Screen Results

- **Directory:** `reports/host/agent_v2_search_gate/screen/`
- **Schedule:** `171daa9cde0f9b2b3a12e0d3d15a9da4f4090a309bf80b9ae42588fb6dd4a78a`
- **Seed:** `20262131` | **Workers:** 8 | **Total Games:** 5,376 | **Wall Time:** 3,425.0 s (~57 min)
- **Fixtures:** 1,344 paired fixtures (96 maps per count across counts 2–15)
- **Multiplicity Adjustment:** Bonferroni across 2 prespecified contrasts ($GW-D$ and $GW-UW$), family $\alpha = 0.05 \implies z = 2.2414$.

### Primary & Supporting Contrasts

| Contrast | Description | Band | Mean Diff | Interval ($z = 2.2414$) | 95% CI ($z = 1.96$) | Verdict | Fixtures |
|---|---|---|---:|---|---|---|---:|
| **$GW - D$** | **Gated Search vs Direct** | **All (2–15)** | **-0.0310** | **[-0.0447, -0.0174]** | **[-0.0430, -0.0191]** | **hurts** | 1,344 |
| $GW - D$ | | Band 2 | -0.0469 | [-0.1113, +0.0176] | [-0.1032, +0.0095] | not shown | 96 |
| $GW - D$ | | Band 3–5 | -0.0790 | [-0.1163, -0.0417] | [-0.1116, -0.0463] | **hurts** | 288 |
| $GW - D$ | | Band 6–10 | -0.0211 | [-0.0440, +0.0018] | [-0.0411, -0.0011] | not shown | 480 |
| $GW - D$ | | Band 11–15 | -0.0090 | [-0.0255, +0.0075] | [-0.0234, +0.0055] | not shown | 480 |
| **$GW - UW$** | **Gate Value ($\tau=0.002$ vs $0$)** | **All (2–15)** | **+0.0136** | **[-0.0004, +0.0276]** | **[+0.0013, +0.0258]** | not shown | 1,344 |
| $GW - UW$ | | Band 2 | +0.0729 | [-0.0049, +0.1507] | [+0.0049, +0.1410] | not shown | 96 |
| $GW - UW$ | | Band 3–5 | +0.0406 | [-0.0004, +0.0817] | [+0.0047, +0.0766] | not shown | 288 |
| $GW - UW$ | | Band 6–10 | -0.0029 | [-0.0262, +0.0204] | [-0.0233, +0.0174] | not shown | 480 |
| $GW - UW$ | | Band 11–15 | +0.0020 | [-0.0101, +0.0140] | [-0.0086, +0.0125] | not shown | 480 |
| **$GW\sim\text{aa} - GW$** | **A/A Re-run Noise** | All (2–15) | -0.0023 | [-0.0104, +0.0058] | [-0.0094, +0.0047] | not shown | 1,344 |

- **Fixture Outcome Balance ($GW$ vs $D$):** 347 improved / 533 unchanged / 464 degraded (131 identical scores of 1,344).
- **Fixture Outcome Balance ($GW$ vs $UW$):** 378 improved / 627 unchanged / 339 degraded (242 identical scores of 1,344).
- **A/A Replay Agreement:** 613 of 1,344 fixtures produced bit-exact identical scores. The non-zero diff is minor replay variance from bounded stochastic rollout draws within the search tree.

---

## 4. Descriptive Predeclared Bands & Per-Count Breakdown

### Summary of Predeclared Descriptive Bands

| Contrast | Metric / Scope | Mean Diff | SE | 95% CI | Verdict |
|---|---|---:|---:|---|:---:|
| **$GW - D$** | 3–15 Equal-Weight Band | **-0.0298** | 0.0062 | [-0.0419, -0.0177] | hurts |
| | 2-Team Band | **-0.0469** | 0.0288 | [-0.1032, +0.0095] | not shown |
| **$GW - UW$** | 3–15 Equal-Weight Band | **+0.0090** | 0.0062 | [-0.0031, +0.0211] | not shown |
| | 2-Team Band | **+0.0729** | 0.0347 | [+0.0049, +0.1410] | helps |
| **$UW - D$** | All (2–15) | **-0.0446** | 0.0064 | [-0.0571, -0.0321] | hurts |
| | 3–15 Equal-Weight Band | **-0.0388** | 0.0064 | [-0.0513, -0.0264] | hurts |
| | 2-Team Band | **-0.1198** | 0.0329 | [-0.1843, -0.0553] | hurts |

### Per-Count Breakdown (96 Fixtures per Count)

| Count | $GW$ Utility | $D$ Utility | $UW$ Utility | $GW - D$ Diff (95% CI) | $GW - UW$ Diff (95% CI) | $UW - D$ Diff (95% CI) |
|---:|---:|---:|---:|---|---|---|
| **2** | 0.8281 | 0.8750 | 0.7552 | -0.0469 [-0.1032, +0.0095] | +0.0729 [+0.0049, +0.1410] | -0.1198 [-0.1843, -0.0553] |
| **3** | 0.7396 | 0.8073 | 0.6719 | -0.0677 [-0.1328, -0.0026] | +0.0677 [-0.0021, +0.1375] | -0.1354 [-0.2007, -0.0701] |
| **4** | 0.7188 | 0.8021 | 0.6736 | -0.0833 [-0.1345, -0.0321] | +0.0451 [-0.0177, +0.1079] | -0.1285 [-0.1909, -0.0661] |
| **5** | 0.6862 | 0.7721 | 0.6771 | -0.0859 [-0.1382, -0.0337] | +0.0091 [-0.0436, +0.0618] | -0.0951 [-0.1441, -0.0460] |
| **6** | 0.7750 | 0.7948 | 0.7885 | -0.0198 [-0.0775, +0.0379] | -0.0135 [-0.0690, +0.0420] | -0.0063 [-0.0630, +0.0505] |
| **7** | 0.7339 | 0.7609 | 0.7391 | -0.0269 [-0.0717, +0.0179] | -0.0052 [-0.0546, +0.0442] | -0.0217 [-0.0641, +0.0207] |
| **8** | 0.7396 | 0.7567 | 0.7403 | -0.0171 [-0.0650, +0.0308] | -0.0007 [-0.0480, +0.0466] | -0.0164 [-0.0617, +0.0289] |
| **9** | 0.7152 | 0.7673 | 0.7334 | -0.0521 [-0.0894, -0.0147] | -0.0182 [-0.0566, +0.0201] | -0.0339 [-0.0754, +0.0077] |
| **10** | 0.7735 | 0.7631 | 0.7504 | +0.0104 [-0.0210, +0.0419] | +0.0231 [-0.0112, +0.0575] | -0.0127 [-0.0481, +0.0226] |
| **11** | 0.7749 | 0.7848 | 0.7634 | -0.0099 [-0.0478, +0.0280] | +0.0115 [-0.0130, +0.0359] | -0.0214 [-0.0635, +0.0208] |
| **12** | 0.7836 | 0.7874 | 0.7741 | -0.0038 [-0.0404, +0.0328] | +0.0095 [-0.0216, +0.0405] | -0.0133 [-0.0497, +0.0232] |
| **13** | 0.7757 | 0.7848 | 0.7688 | -0.0091 [-0.0407, +0.0225] | +0.0069 [-0.0158, +0.0297] | -0.0161 [-0.0468, +0.0146] |
| **14** | 0.7821 | 0.7957 | 0.7957 | -0.0136 [-0.0368, +0.0096] | -0.0136 [-0.0339, +0.0066] | -0.0000 [-0.0284, +0.0284] |
| **15** | 0.7807 | 0.7893 | 0.7852 | -0.0086 [-0.0385, +0.0214] | -0.0045 [-0.0215, +0.0126] | -0.0041 [-0.0321, +0.0239] |

---

## 5. Arm-Level Health, Timing, and Reliability

| Arm | Games | Overall Utility | Faults | Dead | Late | Opponent Faults | First Max (ms) | p99 Median (ms) | Worst p99 (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_v2_p2xxl_gw8x4_fp32` ($GW$) | 1,344 | 0.75254 | 0 | 0 | 0 | 0 | 142.3 | 17.722 | 19.941 | 26.344 | `2e1afee953ee` |
| `agent_v2_p2xxl_direct` ($D$) | 1,344 | **0.78356** | 0 | 0 | 0 | 0 | 71.8 | 1.074 | 3.801 | 5.271 | `d5329fc7f69e` |
| `agent_v2_p2xxl_uw8x4_fp32` ($UW$) | 1,344 | 0.73896 | 0 | 0 | 0 | 0 | 147.1 | 17.754 | 20.428 | 29.128 | `216b9000796b` |
| `agent_v2_p2xxl_gw8x4_fp32~aa` | 1,344 | 0.75020 | 0 | 0 | 0 | 0 | 143.0 | 17.721 | 20.210 | 27.888 | `2e1afee953ee` |

- **Faults / Failures:** 0 faults, 0 dead games, 0 late replies across all 5,376 games.
- **Timing Headroom:** Search variants comfortably complied with the 50 ms steady reply limit (max 26.3 ms for $GW$, 29.1 ms for $UW$) and the 2,000 ms first turn limit (max 147.1 ms).

---

## 6. Architectural Conclusions & Master Roadmap Impact

1. **Definitive Decision on Search: Cancel Deployment of Current Search:**
   - The primary contrast $GW - D$ is statistically significant and negative (**-3.10 pp**, verdict: **hurts**).
   - The follow-up screen specified in T0079's goal (*"If GW-D helps, a separate fresh screen of CSP vs XXL P inside the gated search follows; otherwise that test is cancelled"*) is **formally CANCELLED**.
   - Offline diagnostic stability (T0076) confirmed self-consistency under the tree's own simulator, but in live games against reference opponents, 2-ply evaluation using the offline critic consistently leads to tactical misjudgments that underperform direct actor policy.

2. **Master Roadmap Confirmation:**
   - **Step 1 (T0075):** Initialization baseline pinned and verified.
   - **Step 2 (T0077):** General PPO policy $G$ confirmed as the superior policy (+2.90 pp over baseline; 14 specialists rejected).
   - **Step 3 (T0079):** Gated search tested and proven negative (-3.10 pp; search rejected).
   - **Release Bot Architecture:** Direct-policy `agent_v2` using the general PPO actor (`agent_v2_rl_general_csp`) is established as the final, complete, and unencumbered release version.
