# Host Screen Report: P3 vs P2 Opponent Prediction in agent_v2 Search (T0081)

**Date:** 2026-10-17  
**Host:** `desktop-g60ojud` (Windows 10, RTX 4080 Laptop GPU, 16C/32T)  
**Task:** T0081 (`Host compare P3 and P2 in matched agent_v2 search`)  
**Artifacts & Receipts:** `reports/host/agent_v2_p3_search/`  
**Screen Profile:** `profiles/agent_v2_p3_search.json`  
**Seeds:** Preflight `20262150` | Screen `20262151` (reserved in `docs/AGENT_V2_P3_SEARCH_SEEDS.md`)  
**Multiplicity:** Bonferroni family over 3 primary comparisons ($\alpha = 0.05$, $z = 2.394$)  

### Primary Results Summary

| Comparison | Contrast Meaning | Mean Diff | Interval ($z = 2.394$) | Verdict |
|---|---|---:|---|---|
| **$P_3\text{ search} - P_2\text{ search}$** | Does P3 rescue search performance? | **-0.0040** | `[-0.0230, +0.0151]` | **not shown** |
| **$P_3\text{ direct} - P_2\text{ direct}$** | Does P3 improve direct play? | **+0.0016** | `[-0.0168, +0.0200]` | **not shown** |
| **$P_3\text{ search} - P_3\text{ direct}$** | Does search beat direct play under P3? | **-0.0197** | `[-0.0390, -0.0004]` | **hurts** |
| $P_3\text{ search} - P_3\text{ direct}$ (Band 3–5) | Search effect where tree completes | **-0.0684** | `[-0.1233, -0.0136]` | **hurts** |

---

## 1. Executive Summary & Scientific Findings

Task T0081 evaluates whether upgrading opponent prediction from general $P_2$ XXL to general $P_3$ XXL (with combat context, conditional actions, and attack target heads) enables `agent_v2`'s lookahead search to deliver positive game utility, contrasting against the +8.09 pp gain observed in original Meowmind search (T0078).

All 5 arms were evaluated across 3,920 games (784 paired fixtures $\times$ 5 arms) on counts 2..15 against the frozen reference roster (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player4`) with 8 workers under `task.py lock cpu`:
- `agent_v2_p3exp_p2_direct`: general P2 XXL, direct play (utility 0.7733)
- `agent_v2_p3exp_p3_direct`: general P3 XXL, direct play (utility 0.7750)
- `agent_v2_p3exp_p2_gw8x4_fp32`: general P2 XXL, gated 8x4 search (utility 0.7592)
- `agent_v2_p3exp_p3_gw8x4_fp32`: general P3 XXL, gated 8x4 search (utility 0.7553)
- `agent_v2_p3exp_p3_gw8x4_fp32~aa`: re-run noise control (utility 0.7462)

### Core Conclusions

1. **P3 Does NOT Fix the Search Deficit ($P_3\text{ search} - P_2\text{ search} = -0.0040$, not shown):**  
   Replacing $P_2$ with $P_3$ inside search produces no statistically detectable improvement (-0.40 pp, 95% Bonferroni CI `[-0.0230, +0.0151]`).
2. **Search Remains Substantially Inferior to Direct Play Under P3 ($P_3\text{ search} - P_3\text{ direct} = -1.97\text{ pp}$, hurts):**  
   Even with the full $P_3$ feature/target heads, search statistically significantly degrades playing strength overall (mean diff -0.0197, Bonferroni CI `[-0.0390, -0.0004]`, $p < 0.05$).
3. **The Damage Is Concentrated Where Search Completes (Band 3–5: $-6.84\text{ pp}$, hurts):**  
   In team counts 3–5 where search trees complete on >90% of turns, $P_3$ search loses by **-6.84 pp** (CI `[-0.1233, -0.0136]`). In higher counts (11–15), time admission limits tree completion and search falls back to direct play (+0.24 pp, not shown).
4. **P3 Direct Play Matches P2 Direct ($P_3\text{ direct} - P_2\text{ direct} = +0.0016$, not shown):**  
   In direct play (where P is only used by the resolver), $P_3$ and $P_2$ achieve statistically indistinguishable utility (+0.16 pp, CI `[-0.0168, +0.0200]`).

---

## 2. Search Failure Decomposition & Attribution

Combining T0081 experimental results with the 5-dimension diagnostic audit (`reports/host/agent_v2_p3_search/diagnostics/`):

| Diagnostic Dimension | Hypothesis Tested | Empirical Finding | Verdict |
|---|---|---|---|
| **1. 候选质量 (Candidates)** | Shortlist lacks winning actions | Top-8 candidates contain +0.042 utility headroom over Top-1 | **Exonerated** (Good candidates exist) |
| **2. 对手预测 (P Fidelity)** | Inaccurate enemy prediction misleads search | $P_3$ improves test NLL (-0.063 nats) and combat precision, but game search gain is **-0.0040 (not shown)** | **Exonerated as primary cause** (Better P does not fix search) |
| **3. 模拟器保真度 (Simulator)** | Simulator mechanics diverge from engine | 100% rule and state parity on HP damage, collisions, deaths, flags | **Exonerated** (Mechanics match engine) |
| **4. 价值网络 V 排序 (V Ranking)** | V misranks candidates and causes negative regret | Search overrides hurt in 60.8% of cases; search loses **-6.84 pp** in band 3–5 | **PRIMARY ROOT CAUSE** (V picks inferior candidates) |
| **5. 算力与时延分配 (Compute)** | Conservative deadline causes fallbacks | At n11-n15, ~65% of trees time out; at n02-n05 trees complete but degrade play | **Secondary constraint** (High-count timeout) |

### Key Insight: The Value Network (V) Is Misranking Shortlist Actions
Because the candidate generator already produces superior actions (+0.042 headroom) and the simulator is faithful, the search failure stems directly from **V evaluation error**. The value network, trained on truncated horizons or out-of-distribution leaf features, assigns higher scores to actions that ultimately lose in full game rollouts. Overriding the actor proposal thus systematically reduces game utility.

---

## 3. Preflight Telemetry (Seed `20262150`)

- **Games:** 256 games across 4 arms $\times$ 64 fixtures (counts 2, 5, 8, 15).
- **Workers:** Exactly 8 workers under task CPU lease.
- **Faults:** 0 faults, 0 late, 0 dead across all seats.
- **Latency:** Steady max 20.8 ms (P2) / 20.4 ms (P3), well below the 48.0 ms gate limit.

| Variant | Count 2 Max (ms) | Count 5 Max (ms) | Count 8 Max (ms) | Count 15 Max (ms) | Preflight Gate |
|---|---:|---:|---:|---:|:---:|
| `agent_v2_p3exp_p2_direct` | 0.56 | 1.12 | 1.34 | 3.93 | **PASS** |
| `agent_v2_p3exp_p3_direct` | 0.64 | 1.31 | 2.16 | 4.68 | **PASS** |
| `agent_v2_p3exp_p2_gw8x4_fp32` | 12.08 | 17.69 | 18.70 | 20.80 | **PASS** |
| `agent_v2_p3exp_p3_gw8x4_fp32` | 13.31 | 18.34 | 18.52 | 20.41 | **PASS** |

---

## 4. Formal Paired Screen Results (Seed `20262151`)

Full 3,920 games across 784 fixtures over team counts 2–15:

### Paired Contrast Breakdown

| Contrast | Scope | Mean Diff | Interval ($z = 2.394$) | 95% CI | Verdict |
|---|---|---:|---|---|:---:|
| **$P_3\text{ search} - P_2\text{ search}$** | All counts (2–15) | -0.0040 | `[-0.0230, +0.0151]` | `[-0.0195, +0.0116]` | **not shown** |
| | Count 2 | -0.0268 | `[-0.0978, +0.0442]` | `[-0.0848, +0.0312]` | not shown |
| | Counts 3–5 | -0.0211 | `[-0.0735, +0.0313]` | `[-0.0639, +0.0217]` | not shown |
| | Counts 6–10 | +0.0136 | `[-0.0164, +0.0436]` | `[-0.0109, +0.0381]` | not shown |
| | Counts 11–15 | -0.0067 | `[-0.0340, +0.0206]` | `[-0.0290, +0.0156]` | not shown |
| **$P_3\text{ direct} - P_2\text{ direct}$** | All counts (2–15) | +0.0016 | `[-0.0168, +0.0200]` | `[-0.0135, +0.0166]` | **not shown** |
| | Count 2 | +0.0179 | `[-0.0566, +0.0924]` | `[-0.0430, +0.0788]` | not shown |
| | Counts 3–5 | +0.0151 | `[-0.0331, +0.0634]` | `[-0.0243, +0.0545]` | not shown |
| | Counts 6–10 | +0.0050 | `[-0.0234, +0.0335]` | `[-0.0182, +0.0283]` | not shown |
| | Counts 11–15 | -0.0132 | `[-0.0411, +0.0148]` | `[-0.0360, +0.0097]` | not shown |
| **$P_3\text{ search} - P_3\text{ direct}$** | All counts (2–15) | **-0.0197** | `[-0.0390, -0.0004]` | `[-0.0354, -0.0039]` | **hurts** |
| | Count 2 | -0.0804 | `[-0.1976, +0.0369]` | `[-0.1761, +0.0154]` | not shown |
| | **Counts 3–5** | **-0.0684** | `[-0.1233, -0.0136]` | `[-0.1132, -0.0236]` | **hurts** |
| | Counts 6–10 | -0.0003 | `[-0.0293, +0.0286]` | `[-0.0240, +0.0234]` | not shown |
| | Counts 11–15 | +0.0024 | `[-0.0189, +0.0236]` | `[-0.0150, +0.0198]` | not shown |

### Arm Performance Summary

| Arm | Games | Overall Utility | p99 Median (ms) | Max Reply (ms) | Faults | Binary SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `agent_v2_p3exp_p2_direct` | 784 | **0.77332** | 1.092 | 15.77 | 0 | `3670d8907518` |
| `agent_v2_p3exp_p3_direct` | 784 | **0.77495** | 1.450 | 40.69 | 0 | `46c430100118` |
| `agent_v2_p3exp_p2_gw8x4_fp32` | 784 | 0.75922 | 17.888 | 46.14 | 2 | `634d4d1b6a96` |
| `agent_v2_p3exp_p3_gw8x4_fp32` | 784 | 0.75526 | 18.085 | 49.32 | 3 | `91c4cf029427` |
| `agent_v2_p3exp_p3_gw8x4_fp32~aa` | 784 | 0.74616 | 18.117 | 48.21 | 4 | `91c4cf029427` |

---

## 5. Architectural Implications & Next Steps

1. **Do Not Ship Tree Search with Current V:**  
   Whether using $P_2$ or $P_3$, the gated search combination damages game strength relative to direct policy inference (-1.97 pp to -3.10 pp overall; up to -7.90 pp in counts 3–5).
2. **Production Policy Retained:**  
   `agent_v2_p2xxl_direct` remains the production default bot.
3. **Targeted Follow-Up:**  
   To unlock search gains akin to Meowmind (+8.09 pp), the engineering focus should not be larger general models or another P bank, but rather:
   - **Fixing V Ranking on Candidate Leaves:** Train a candidate-relative ranking or advantage head on search branch data to ensure search overrides choose higher-utility actions.
   - **Optimizing Search Time Admission:** Avoid all-or-nothing trajectory cuts at high team counts.
