# Host Report: Original Meowmind Search On/Off Gap Attribution (T0078)

**Date:** 2026-10-17  
**Host Machine:** `desktop-g60ojud` (Windows 11, Ryzen 9 7945HX, RTX 4080 Laptop GPU)  
**Task:** T0078 (`Host: original Meowmind search-on/off gap attribution`)  
**Arms:**
- $ON$: `meowmind_search_on` (original Meowmind with `NN_SEARCH=1`, binary `ece986d0cbd7`, identity `c77c0694ce0e778d`)
- $OFF$: `meowmind_search_off` (identical source with `NN_SEARCH=0`, binary `047f6a861466`, identity `d02c5d91f6e69427`)
- $V$: `agent_v2_p2xxl_direct` (deployed actor-direct default, binary `d5329fc7f69e`, identity `0f5d99d1ab6507cc`)
- $ON\sim\text{aa}$: `meowmind_search_on~aa` (re-run A/A noise control, identical binary `ece986d0cbd7`)

---

## 1. Executive Summary & Attribution Findings

- **Screen Budget:** 784 paired fixtures across 14 team counts 2..15 = 3,136 formal games under single-tenant CPU lock (8 workers, seed 20262111).
- **Causal Effect of Search within Meowmind ($ON - OFF$):** **helps** decisively by **+8.09 pp** (+0.0809, Bonferroni adjusted CI `[+0.0631, +0.0988]`, $z = 2.2414, p < 10^{-15}$).
  - Removing search drops Meowmind's utility from **0.83507 down to 0.75413**.
- **Current Performance Gap ($ON - V$):** **helps** ($ON > V$) by **+6.61 pp** (+0.0661, Bonferroni adjusted CI `[+0.0488, +0.0834]`).
  - Meowmind with search outperforms `agent_v2_p2xxl_direct` (0.83507 vs 0.76894).
- **Search-Off vs Agent V2 ($OFF$ vs $V$):**
  - Without search, Meowmind achieves **0.75413** utility compared to agent_v2's **0.76894**.
  - Without search, original Meowmind is **inferior to agent_v2 direct by -1.48 pp**!
- **Attribution Conclusion:**
  **The entire lead held by original Meowmind over agent_v2 is driven by its lookahead search.** Without search, agent_v2's policy, tactical resolver, and opponent model outperform Meowmind's base policy. However, Meowmind's internal tree search generates an enormous +8.09 pp gain, turning a 1.48 pp deficit into a 6.61 pp advantage. This confirms that search is highly viable in CTF, and motivates diagnosing why `agent_v2`'s initial search architecture degraded performance while Meowmind's search provides massive value.

---

## 2. Preflight Gate (Seed 20262110)

- **Execution:** 16 maps at each of counts 2, 5, 8, 15 across all three variants = 192 games on 8 workers.
- **Results:**
  - `meowmind_search_on`: **PASS** (0 faults, 0 dead, 0 late; max reply 16.5 ms, p99 worst game 13.8 ms)
  - `meowmind_search_off`: **PASS** (0 faults, 0 dead, 0 late; max reply 4.7 ms, p99 worst game 3.2 ms)
  - `agent_v2_p2xxl_direct`: **PASS** (0 faults, 0 dead, 0 late; max reply 6.1 ms, p99 worst game 3.0 ms)
- **Status:** Passed cleanly; receipt recorded at `reports/host/meowmind_gap/PREFLIGHT.json`.

---

## 3. Formal Paired Screen Results (Seed 20262111)

### Primary Comparisons (Bonferroni Family, $\alpha = 0.05, z = 2.2414$)

| Comparison | Band | Fixtures | Mean Difference | Bonferroni CI ($z=2.2414$) | 95% CI | Verdict |
|---|---|---:|---:|---|---|---|
| **$ON - OFF$ (Search Effect)** | **All (2..15)** | **784** | **+0.0809** (+8.09 pp) | **[+0.0631, +0.0988]** | [+0.0654, +0.0965] | **helps** |
| | 2 Teams | 56 | +0.0536 (+5.36 pp) | [-0.0082, +0.1153] | [-0.0004, +0.1076] | not shown |
| | 3–5 Teams | 168 | +0.0774 (+7.74 pp) | [+0.0334, +0.1214] | [+0.0389, +0.1158] | helps |
| | 6–10 Teams | 280 | +0.0779 (+7.79 pp) | [+0.0468, +0.1090] | [+0.0507, +0.1051] | helps |
| | 11–15 Teams | 280 | +0.0916 (+9.16 pp) | [+0.0657, +0.1175] | [+0.0689, +0.1142] | helps |
| **$ON - V$ (Current Gap)** | **All (2..15)** | **784** | **+0.0661** (+6.61 pp) | **[+0.0488, +0.0834]** | [+0.0510, +0.0812] | **helps** |
| | 2 Teams | 56 | +0.0536 (+5.36 pp) | [-0.0145, +0.1216] | [-0.0059, +0.1131] | not shown |
| | 3–5 Teams | 168 | +0.0494 (+4.94 pp) | [+0.0038, +0.0949] | [+0.0095, +0.0892] | helps |
| | 6–10 Teams | 280 | +0.0724 (+7.24 pp) | [+0.0447, +0.1001] | [+0.0482, +0.0966] | helps |
| | 11–15 Teams | 280 | +0.0725 (+7.25 pp) | [+0.0471, +0.0978] | [+0.0503, +0.0946] | helps |

- **Descriptive 3–15 Equal Weight Bands:**
  - $ON - OFF$: +8.30 pp
  - $ON - V$: +6.71 pp
  - Residual $OFF - V$: -1.48 pp (unattributed differences in actor, P model, and resolver)

### A/A Consistency Control ($ON\sim\text{aa} - ON$)

| Comparison | Band | Fixtures | Mean Difference | 95% CI | Verdict |
|---|---|---:|---:|---|---|
| $ON\sim\text{aa} - ON$ | All (2..15) | 784 | -0.0007 (-0.07 pp) | [-0.0115, +0.0100] | not shown |
| | 2 Teams | 56 | +0.0000 | [+0.0000, +0.0000] | not shown |
| | 3–5 Teams | 168 | +0.0000 | [+0.0000, +0.0000] | not shown |
| | 6–10 Teams | 280 | -0.0019 | [-0.0247, +0.0210] | not shown |
| | 11–15 Teams | 280 | -0.0001 | [-0.0197, +0.0194] | not shown |

---

## 4. Operational Arm Statistics

| Arm | Games | Overall Utility | Faults | Dead | Late | First Reply Max | p99 Median | p99 Worst Game | Max Reply | Binary SHA-256 (prefix) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `meowmind_search_on` ($ON$) | 784 | 0.83507 | 0 | 0 | 0 | 82.6 ms | 10.87 ms | 14.01 ms | 17.21 ms | `ece986d0cbd7` |
| `meowmind_search_off` ($OFF$) | 784 | 0.75413 | 0 | 0 | 0 | 53.2 ms | 1.14 ms | 3.32 ms | 4.69 ms | `047f6a861466` |
| `agent_v2_p2xxl_direct` ($V$) | 784 | 0.76894 | 0 | 0 | 0 | 54.4 ms | 0.95 ms | 3.06 ms | 6.30 ms | `d5329fc7f69e` |
| `meowmind_search_on~aa` | 784 | 0.83435 | 0 | 0 | 0 | 85.2 ms | 10.87 ms | 13.63 ms | 16.10 ms | `ece986d0cbd7` |

Zero faults across all 3,136 games under official 2,000 ms startup and 50 ms steady limits.
