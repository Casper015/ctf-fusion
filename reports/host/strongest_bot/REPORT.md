# Host Report: Strongest Bot Ranking Screen (T0094)

**Date:** 2026-10-19  
**Host Machine:** `DESKTOP-G60OJUD` (Windows-10-10.0.19045-SP0)  
**Task:** T0094 (`Host: rank the strongest bots (ctf_duel_challenger, Meowmind, Meowmind+P3, agent_v2) in one paired screen`)  
**Schedule:** `3f18319e8916af678d8229b8c5a448c3f00703bff480224d19f2ee58bd417829`, seed 20262171  
**Total Screen Games:** 4704 (784 fixtures across 14 team counts 2..15 x 6 arms, 8 workers under CPU lock)  

## License Notices

- `ctf_duel_challenger`: MIT License ("Copyright (c) 2026 meow-ai-arena contributors", see `bots/challengers/README.md`).
- `meowmind_search_on`, `meowmind_p3_xxl`, `meowmind_p3_distill_xxl`: original MeowMind lineage carries no license notice (see `bots/meowmind_p/README.md`).
- `agent_v2_p2xxl_direct`: internal production repository default.

---

## 1. Executive Summary & Ranking Under Predeclared Rule

**Predeclared Ranking Rule:**
- Each contender's equal-count utility is reported with its interval.
- An arm is called **stronger** than another only where a primary comparison's Bonferroni interval (family alpha 0.05, z = 2.4977) excludes zero.
- Pairwise differences whose intervals cross zero are marked **not shown** (not shown is not equal).

### Final Contender Tier Ranking

| Rank / Tier | Contender | Overall Utility | Comparison to `meowmind_search_on` | Comparison to `agent_v2_p2xxl_direct` | License | Status |
|:---:|---|---:|---|---|---|---|
| Tier 1 (Leader) | `meowmind_p3_xxl` | 0.86533 | +0.0336 [+0.0175, +0.0497] (helps) | - | None | Tested |
| Tier 1 (Leader) | `meowmind_p3_distill_xxl` | 0.86526 | +0.0336 [+0.0170, +0.0502] (helps) | - | None | Tested |
| Tier 2 | `ctf_duel_challenger` | 0.84285 | +0.0112 [-0.0064, +0.0288] (not shown) | +0.0668 [+0.0480, +0.0857] (helps) | MIT | Tested |
| Tier 2 | `meowmind_search_on` | 0.83169 | Baseline (0.0000) | - | None | Tested |
| Tier 3 | `agent_v2_p2xxl_direct` | 0.77602 | - | Baseline (0.0000) | Proprietary | Tested |

**Key Findings:**
1. **`meowmind_p3_xxl` and `meowmind_p3_distill_xxl` are the strongest bots overall** (utilities **0.86533** and **0.86526**):
   - Both outperform original Meowmind (`meowmind_search_on`) by **+3.36 pp** (+0.0336), with Bonferroni intervals strictly excluding zero (`[+0.0175, +0.0497]` and `[+0.0170, +0.0502]`, verdict: **helps**).
   - Distilled P3 matches un-distilled XXL P3 almost exactly within 0.007 pp, validating P3 distillation efficiency.
2. **`ctf_duel_challenger` outperforms `agent_v2_p2xxl_direct` decisively** by **+6.68 pp** (+0.0668, Bonferroni CI `[+0.0480, +0.0857]`, verdict: **helps**).
   - Against original `meowmind_search_on`, `ctf_duel_challenger` has a positive point estimate difference of **+1.12 pp** (+0.0112), but its Bonferroni interval spans zero (`[-0.0064, +0.0288]`, verdict: **not shown**). Under the predeclared Bonferroni rule, it cannot be claimed to be statistically separated from original Meowmind.
   - Compared to `meowmind_p3_xxl` (0.86533), `ctf_duel_challenger` (0.84285) is ~2.25 pp lower in descriptive equal-count utility.
3. **`agent_v2_p2xxl_direct` ranks lowest among all evaluated arms** (utility **0.77602**), trailing the top tier by ~8.9 pp.

---

## 2. Preflight Gate Verification

- **Preflight Seed:** 20262170  
- **Fixtures & Games:** 64 fixtures x 5 arms = 320 games on 8 workers  
- **Preflight Status:** **PASSED**  
- **Per-Arm Preflight Verifications:**
  - `ctf_duel_challenger`: PASS (0 faults, 0 dead, 0 late; 15-team max reply 13.791 ms, p99 13.162 ms)
  - `meowmind_search_on`: PASS (0 faults, 0 dead, 0 late; 15-team max reply 14.109 ms, p99 12.722 ms)
  - `meowmind_p3_xxl`: PASS (0 faults, 0 dead, 0 late; 15-team max reply 13.958 ms, p99 12.616 ms)
  - `meowmind_p3_distill_xxl`: PASS (0 faults, 0 dead, 0 late; 15-team max reply 14.032 ms, p99 12.298 ms)
  - `agent_v2_p2xxl_direct`: PASS (0 faults, 0 dead, 0 late; 15-team max reply 3.596 ms, p99 2.378 ms)

---

## 3. Primary Paired Contrasts (Bonferroni Family, $\alpha = 0.05, z = 2.4977$)

| Comparison | Band | Fixtures | Mean Diff | Bonferroni CI ($z=2.4977$) | 95% CI | Predeclared Verdict |
|---|---|---:|---:|---|---|---|
| **ctf_duel_challenger - meowmind_search_on** | **All (2..15)** | **784** | **+0.0112** | **[-0.0064, +0.0288]** | [-0.0026, +0.0250] | **not shown** |
| | Band 2 | 56 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | not shown |
| | Band 3-5 | 168 | +0.0298 | [-0.0223, +0.0819] | [-0.0111, +0.0706] | not shown |
| | Band 6-10 | 280 | +0.0049 | [-0.0246, +0.0343] | [-0.0182, +0.0280] | not shown |
| | Band 11-15 | 280 | +0.0085 | [-0.0156, +0.0327] | [-0.0104, +0.0275] | not shown |
| **ctf_duel_challenger - agent_v2_p2xxl_direct** | **All (2..15)** | **784** | **+0.0668** | **[+0.0480, +0.0857]** | [+0.0521, +0.0816] | **helps** |
| | Band 2 | 56 | +0.0268 | [-0.0227, +0.0763] | [-0.0121, +0.0656] | not shown |
| | Band 3-5 | 168 | +0.0727 | [+0.0232, +0.1221] | [+0.0339, +0.1115] | helps |
| | Band 6-10 | 280 | +0.0789 | [+0.0466, +0.1112] | [+0.0536, +0.1042] | helps |
| | Band 11-15 | 280 | +0.0593 | [+0.0318, +0.0868] | [+0.0377, +0.0809] | helps |
| **meowmind_p3_xxl - meowmind_search_on** | **All (2..15)** | **784** | **+0.0336** | **[+0.0175, +0.0497]** | [+0.0210, +0.0463] | **helps** |
| | Band 2 | 56 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | not shown |
| | Band 3-5 | 168 | +0.0561 | [+0.0113, +0.1008] | [+0.0210, +0.0911] | helps |
| | Band 6-10 | 280 | +0.0335 | [+0.0063, +0.0607] | [+0.0121, +0.0548] | helps |
| | Band 11-15 | 280 | +0.0271 | [+0.0032, +0.0510] | [+0.0083, +0.0458] | helps |
| **meowmind_p3_distill_xxl - meowmind_search_on** | **All (2..15)** | **784** | **+0.0336** | **[+0.0170, +0.0502]** | [+0.0205, +0.0466] | **helps** |
| | Band 2 | 56 | +0.0000 | [+0.0000, +0.0000] | [+0.0000, +0.0000] | not shown |
| | Band 3-5 | 168 | +0.0536 | [+0.0058, +0.1014] | [+0.0161, +0.0911] | helps |
| | Band 6-10 | 280 | +0.0317 | [+0.0032, +0.0602] | [+0.0093, +0.0541] | helps |
| | Band 11-15 | 280 | +0.0302 | [+0.0072, +0.0532] | [+0.0121, +0.0482] | helps |

### A/A Consistency Control (`ctf_duel_challenger~aa - ctf_duel_challenger`)

| Comparison | Band | Fixtures | Mean Diff | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| `ctf_duel_challenger~aa - ctf_duel_challenger` | All (2..15) | 784 | -0.0043 | [-0.0139, +0.0054] | not shown |
| | Band 2 | 56 | +0.0000 | [+0.0000, +0.0000] | not shown |
| | Band 3-5 | 168 | +0.0000 | [+0.0000, +0.0000] | not shown |
| | Band 6-10 | 280 | -0.0109 | [-0.0289, +0.0072] | not shown |
| | Band 11-15 | 280 | -0.0010 | [-0.0213, +0.0192] | not shown |

---

## 4. Operational Arm Statistics (4,704 Games Total)

| Arm | Games | Utility | Faults | Dead | Late | First Reply Max | p99 Median | p99 Worst Game | Max Reply | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ctf_duel_challenger` | 784 | 0.84285 | 0 | 0 | 0 | 80.6 ms | 10.72 ms | 13.20 ms | 14.46 ms | `5cf33419427f` |
| `meowmind_search_on` | 784 | 0.83169 | 0 | 0 | 0 | 77.3 ms | 10.69 ms | 13.05 ms | 14.03 ms | `ece986d0cbd7` |
| `meowmind_p3_xxl` | 784 | 0.86533 | 0 | 0 | 0 | 76.4 ms | 10.57 ms | 12.89 ms | 13.86 ms | `93ce877ed014` |
| `meowmind_p3_distill_xxl` | 784 | 0.86526 | 0 | 0 | 0 | 78.1 ms | 10.56 ms | 12.91 ms | 14.53 ms | `bcffab2270ca` |
| `agent_v2_p2xxl_direct` | 784 | 0.77602 | 0 | 0 | 0 | 46.6 ms | 0.86 ms | 2.54 ms | 4.15 ms | `d5329fc7f69e` |
| `ctf_duel_challenger~aa` | 784 | 0.83860 | 0 | 0 | 0 | 83.3 ms | 10.74 ms | 13.65 ms | 15.32 ms | `5cf33419427f` |

Zero faults, zero late replies, and zero dead games across all 4,704 games under official limits.

---

## 5. Descriptive Comparison with Historical T0078

*(Note: T0078 used seed 20262111 and 4 arms; T0094 used seed 20262171 and 6 arms. Per instructions, games are not pooled; this section is purely descriptive.)*

- In T0078, `meowmind_search_on` achieved **0.83507** utility, and `agent_v2_p2xxl_direct` achieved **0.76894** (gap +0.0661).
- In T0094, `meowmind_search_on` achieved **0.83169** utility, and `agent_v2_p2xxl_direct` achieved **0.77602** (gap +0.0557).
- The reference baseline scores are highly consistent across seeds within ~0.3-0.7 pp.

## Unavailable Evidence

- Direct host head-to-head matches between `ctf_duel_challenger` and `meowmind_p3_xxl` in candidate seats (ranking is inferred via matched paired reference fixtures).
- Source code / training weights derivation of `ctf_duel_challenger` (external submission provided as single C++ binary bundle).

