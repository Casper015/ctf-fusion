# Host Screen Report: Agent V2 RL Policy Acceptance Screen (T0077)

**Date:** 2026-10-17  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0077 (`Host: exact-count RL specialists versus general control and common initialization`)  
**Artifacts & Receipts:** `reports/host/agent_v2_rl_acceptance/`  
**Screen Profile:** `profiles/agent_v2_rl_acceptance.json`  
**Seeds:** Preflight `20262100` | Screen `20262101` (reserved in `docs/AGENT_V2_RL_ACCEPTANCE_SEEDS.md`)  
**Bonferroni Multiplicity:** $\alpha = 0.05$, $k = 2$ comparisons ($z = 2.2414$)  
**Primary Contrast ($S - G$):** **hurts** (mean diff = -0.0277, Bonferroni CI [-0.0406, -0.0148])  
**Supporting Contrast ($G - I$):** **helps** (mean diff = +0.0290, Bonferroni CI [+0.0157, +0.0422])  

---

## 1. Executive Summary

Task T0077 executes the formal paired policy acceptance screen for the `agent_v2` carrier, comparing three actor-direct configurations using the identical CSP P bank and extracted resolver:
- **Arm $S$ (`agent_v2_rl_csp`):** The 14 exact-count PPO RL policy specialists from T0068 (binary `458f51f068b2`).
- **Arm $G$ (`agent_v2_rl_general_csp`):** The compute-matched general RL control from T0068 (binary `08d1d5602e4f`).
- **Arm $I$ (`agent_v2_rl_init_csp`):** The common initialization actor from T0075 (binary `7e4280feda0d`).
- **Arm $S\sim\text{aa}$ (`agent_v2_rl_csp~aa`):** Re-run noise control playing identical binary `458f51f068b2`.

The screen answers two core architectural questions:
1. **Does exact-count specialization help? ($S - G$)**  
   **No; it hurts significantly.** Specialists perform **-2.77 pp** worse overall (Bonferroni CI [-0.0406, -0.0148], $p < 0.001$). On counts 3–15, specialists trail the general control by **-2.90 pp** (95% CI [-0.0410, -0.0170]). Specialists degrade performance on 11 of the 14 individual team counts.
2. **Does continuing PPO on the `agent_v2` carrier help? ($G - I$)**  
   **Yes; it helps strongly.** The general PPO policy achieves a statistically significant **+2.90 pp** gain over the common initialization baseline (Bonferroni CI [+0.0157, +0.0422], $p < 0.001$). On counts 3–15, the gain is **+3.24 pp** (95% CI [+0.0201, +0.0447]).

### Training Exposure Context
As documented in `reports/models/count_specialists_rl/REPORT.md`, the general arm ($G$) was trained across 1,400,000 steps on uniformly sampled counts, realizing between **222 and 279 episodes per count** (not strictly 250). The exact-count specialists ($S$) received 100,000 steps exclusively on their single assigned count (~250 episodes each). Despite matched compute per count, the general model benefited dramatically from multi-count data diversity and broad representation learning, whereas single-count specialists suffered from overfitting to narrow rollout distributions.

---

## 2. Eight-Worker Preflight Gate

- **Receipt:** `reports/host/agent_v2_rl_acceptance/PREFLIGHT.json`
- **Seed:** `20262100` | **Workers:** 8 (exact) | **Wall Time:** 17.5 s  
- **Fixtures:** 64 total (16 maps each for counts 2, 5, 8, 15) against reference roster (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player4`)
- **Total Games:** 192 (64 fixtures $\times$ 3 variants)
- **Gates:** first reply < 1,800.0 ms, steady reply < 48.0 ms, 0 all-seat faults/late/dead, 100% telemetry coverage.

| Variant | Role | Games | Turn Coverage | Steady Max (ms) | First Max (ms) | Faults / Late / Dead | Gate Verdict |
|---|---|---:|---:|---:|---:|---:|---|
| `agent_v2_rl_csp` | Arm $S$ (14 specialists) | 64 | 100.0% | 6.42 | 62.7 | 0 / 0 / 0 | **PASS** |
| `agent_v2_rl_general_csp` | Arm $G$ (General control) | 64 | 100.0% | 9.92 | 60.7 | 0 / 0 / 0 | **PASS** |
| `agent_v2_rl_init_csp` | Arm $I$ (Init baseline) | 64 | 100.0% | 5.97 | 54.6 | 0 / 0 / 0 | **PASS** |

All arms met every safety, latency, and telemetry gate under full 8-worker host load.

---

## 3. Formal Paired Screen Results

- **Directory:** `reports/host/agent_v2_rl_acceptance/screen/`
- **Schedule:** `bbab7d975e5c2af6f504b79684afc200ce58d529d4342159aec9fa6f3584aba1`
- **Seed:** `20262101` | **Workers:** 8 | **Total Games:** 5,376 | **Wall Time:** 637.2 s (~10.6 min)
- **Fixtures:** 1,344 paired fixtures (96 maps per count across counts 2–15)
- **Multiplicity Adjustment:** Bonferroni across 2 prespecified contrasts ($S-G$ and $G-I$), family $\alpha = 0.05 \implies z = 2.2414$.

### Primary & Supporting Contrasts

| Contrast | Description | Band | Mean Diff | Interval ($z = 2.2414$) | 95% CI ($z = 1.96$) | Verdict | Fixtures |
|---|---|---|---:|---|---|---|---:|
| **$S - G$** | **Specialization Effect** | **All (2–15)** | **-0.0277** | **[-0.0406, -0.0148]** | **[-0.0390, -0.0164]** | **hurts** | 1,344 |
| $S - G$ | | Band 2 | -0.0104 | [-0.0391, +0.0182] | [-0.0355, +0.0146] | not shown | 96 |
| $S - G$ | | Band 3–5 | -0.0279 | [-0.0630, +0.0072] | [-0.0585, +0.0028] | not shown | 288 |
| $S - G$ | | Band 6–10 | -0.0316 | [-0.0534, -0.0098] | [-0.0506, -0.0125] | **hurts** | 480 |
| $S - G$ | | Band 11–15 | -0.0271 | [-0.0459, -0.0083] | [-0.0435, -0.0107] | **hurts** | 480 |
| **$G - I$** | **PPO Training Gain** | **All (2–15)** | **+0.0290** | **[+0.0157, +0.0422]** | **[+0.0174, +0.0405]** | **helps** | 1,344 |
| $G - I$ | | Band 2 | -0.0156 | [-0.0416, +0.0104] | [-0.0384, +0.0071] | not shown | 96 |
| $G - I$ | | Band 3–5 | +0.0273 | [-0.0073, +0.0620] | [-0.0030, +0.0577] | not shown | 288 |
| $G - I$ | | Band 6–10 | +0.0213 | [-0.0011, +0.0436] | [+0.0017, +0.0408] | not shown | 480 |
| $G - I$ | | Band 11–15 | +0.0465 | [+0.0263, +0.0668] | [+0.0288, +0.0642] | **helps** | 480 |
| **$S\sim\text{aa} - S$** | **A/A Re-run Noise** | All (2–15) | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | not shown | 1,344 |

- **Fixture Outcome Balance ($S$ vs $G$):** 389 improved / 431 unchanged / 524 degraded (1 identical score of 1,344).
- **Fixture Outcome Balance ($G$ vs $I$):** 520 improved / 422 unchanged / 402 degraded (1 identical score of 1,344).
- **A/A Determinism Diagnostic:** 1,344 of 1,344 fixtures produced bit-exact identical final scores (+0.0000 [+0.0000, +0.0000]), demonstrating zero concurrency cross-talk or nondeterministic drift.

---

## 4. Descriptive Predeclared Bands & Per-Count Analysis

Per task specification, 2-team fixtures were saturated vs the reference roster in earlier screens (T0057/T0073). We report the descriptive equal-weight 3–15 band, the 2-team band, and the descriptive $S - I$ contrast:

### Summary of Predeclared Descriptive Bands

| Contrast | Metric / Scope | Mean Diff | SE | 95% CI |
|---|---|---:|---:|---|
| **$S - G$** (Specialization) | 3–15 Equal-Weight Band | **-0.0290** | 0.0061 | [-0.0410, -0.0170] |
| | 2-Team Band | **-0.0104** | 0.0128 | [-0.0355, +0.0146] |
| **$G - I$** (PPO Gain) | 3–15 Equal-Weight Band | **+0.0324** | 0.0063 | [+0.0201, +0.0447] |
| | 2-Team Band | **-0.0156** | 0.0116 | [-0.0384, +0.0071] |
| **$S - I$** (Specialists vs Init) | All (2–15) | **+0.0013** | 0.0059 | [-0.0104, +0.0129] |
| | 3–15 Equal-Weight Band | **+0.0034** | 0.0063 | [-0.0090, +0.0157] |
| | 2-Team Band | **-0.0260** | 0.0155 | [-0.0564, +0.0043] |

### Per-Count Breakdown (96 Fixtures per Count)

| Count | $S$ Utility | $G$ Utility | $I$ Utility | $S - G$ Diff (95% CI) | $G - I$ Diff (95% CI) | $S - I$ Diff (95% CI) |
|---:|---:|---:|---:|---|---|---|
| **2** | 0.8698 | 0.8802 | 0.8958 | -0.0104 [-0.0355, +0.0146] | -0.0156 [-0.0384, +0.0071] | -0.0260 [-0.0564, +0.0043] |
| **3** | 0.7786 | 0.8177 | 0.7630 | -0.0391 [-0.0950, +0.0168] | +0.0547 [-0.0016, +0.1110] | +0.0156 [-0.0410, +0.0722] |
| **4** | 0.7726 | 0.7691 | 0.8003 | +0.0035 [-0.0510, +0.0580] | -0.0312 [-0.0839, +0.0214] | -0.0278 [-0.0900, +0.0345] |
| **5** | 0.7253 | 0.7734 | 0.7148 | -0.0482 [-0.0970, +0.0007] | +0.0586 [+0.0102, +0.1070] | +0.0104 [-0.0434, +0.0642] |
| **6** | 0.7708 | 0.8125 | 0.7760 | -0.0417 [-0.0927, +0.0094] | +0.0365 [-0.0123, +0.0852] | -0.0052 [-0.0488, +0.0384] |
| **7** | 0.7622 | 0.7483 | 0.7595 | +0.0139 [-0.0321, +0.0599] | -0.0113 [-0.0618, +0.0393] | +0.0026 [-0.0370, +0.0422] |
| **8** | 0.7225 | 0.7634 | 0.7359 | -0.0409 [-0.0857, +0.0038] | +0.0275 [-0.0155, +0.0705] | -0.0134 [-0.0551, +0.0283] |
| **9** | 0.7122 | 0.7598 | 0.7363 | -0.0475 [-0.0804, -0.0147] | +0.0234 [-0.0080, +0.0549] | -0.0241 [-0.0616, +0.0134] |
| **10** | 0.7465 | 0.7882 | 0.7581 | -0.0417 [-0.0777, -0.0057] | +0.0301 [-0.0117, +0.0719] | -0.0116 [-0.0498, +0.0267] |
| **11** | 0.7682 | 0.7755 | 0.7292 | -0.0073 [-0.0424, +0.0278] | +0.0464 [+0.0081, +0.0846] | +0.0391 [-0.0011, +0.0792] |
| **12** | 0.7713 | 0.7884 | 0.7486 | -0.0170 [-0.0584, +0.0243] | +0.0398 [-0.0026, +0.0821] | +0.0227 [-0.0156, +0.0610] |
| **13** | 0.7578 | 0.7843 | 0.7474 | -0.0265 [-0.0606, +0.0076] | +0.0369 [+0.0005, +0.0733] | +0.0104 [-0.0289, +0.0497] |
| **14** | 0.7548 | 0.8065 | 0.7300 | -0.0517 [-0.0892, -0.0142] | +0.0765 [+0.0373, +0.1157] | +0.0248 [-0.0155, +0.0652] |
| **15** | 0.7597 | 0.7928 | 0.7597 | -0.0331 [-0.0684, +0.0022] | +0.0331 [-0.0084, +0.0746] | -0.0000 [-0.0378, +0.0378] |

---

## 5. Arm Level Health, Timing, and Reliability

All four arms played 1,344 games under official production limits (2,000 ms first turn, 50 ms steady turns):

| Arm | Games | Overall Utility | Faults | Dead | Late | Opponent Faults | First Max (ms) | p99 Median (ms) | Worst p99 (ms) | Max Reply (ms) | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `agent_v2_rl_csp` ($S$) | 1,344 | 0.76079 | 0 | 0 | 0 | 0 | 71.1 | 1.119 | 5.043 | 12.523 | `458f51f068b2` |
| `agent_v2_rl_general_csp` ($G$) | 1,344 | **0.78849** | 0 | 0 | 0 | 0 | 61.3 | 1.139 | 4.237 | 6.572 | `08d1d5602e4f` |
| `agent_v2_rl_init_csp` ($I$) | 1,344 | 0.75954 | 0 | 0 | 0 | 0 | 64.5 | 1.141 | 4.670 | 16.405 | `7e4280feda0d` |
| `agent_v2_rl_csp~aa` ($S\sim\text{aa}$) | 1,344 | 0.76079 | 0 | 0 | 0 | 0 | 63.0 | 1.120 | 4.373 | 12.705 | `458f51f068b2` |

- **Faults / Failures:** Exactly 0 faults, 0 dead games, and 0 late replies across all 5,376 games.
- **Timing Headroom:** Steady reply max was 16.4 ms (over 3x headroom below the 50 ms limit); first turn max was 71.1 ms (over 25x headroom below 2,000 ms).
- **Binary & Model Integrity:** All binary hashes matched the manifest and preflight records exactly.

---

## 6. Architectural Conclusions & Downstream Impact

1. **Definitive Decision on Model Routing: General Control Wins:**
   - Continuing PPO on the `agent_v2` carrier strongly and reliably improves playing strength (+2.90 pp overall, +3.24 pp on 3–15 counts).
   - Training 14 exact-count specialists completely destroyed this advantage (-2.77 pp compared to general control), reducing overall performance back down to the baseline initialization model ($S - I = +0.13$ pp).
   - **Conclusion:** The 14 exact-count specialists are rejected for deployment. The compute-matched general policy `rl_general` (`agent_v2_rl_general_csp`) is the empirically proven policy default for `agent_v2`.

2. **Why General Policy Triumphed Over Count Specialists:**
   - In CTF matches, combat mechanics, pathfinding, flag priority, and defensive response require diverse situational experience.
   - The general policy experienced 1.4M steps across varying team densities, learning robust tactical behaviors that transfer seamlessly across counts.
   - The specialists, restricted to 100k steps on a single team count, lacked adequate data diversity and overfit to narrow state-action trajectories.

3. **Master Roadmap Impact (Steps 3, 4, 5):**
   - **Step 2 is complete:** T0077 establishes `agent_v2_rl_general_csp` as the baseline carrier policy.
   - **Next Step (Step 3 / T0079):** Screen search v2 under conservative time admission. The search preflight gate must verify completion rates and non-trivial plan alterations before attempting formal search evaluations.
   - **If search does not show significant benefits:** `agent_v2_rl_general_csp` directly serves as the production `agent_v2` release bot.
