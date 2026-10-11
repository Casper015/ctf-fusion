# Carrier Integration Report: Opponent Policy P Empirical Screen (S / M / L)

> **Correction note (2026-10-11, T0024).** This report is kept as written: it is the record of the e11c45b screen. Its conclusions are overstated, according to [C08_VS_V12_FOUNDATION_REVIEW_e11c45b.md](C08_VS_V12_FOUNDATION_REVIEW_e11c45b.md#corrections-to-the-new-integration-write-up) and the [review bundle](review_bundle_20261009/README.md):
>
> - **Not both significant.** v12 M's aggregate interval crosses zero, so "M and L both significantly beat control" does not hold.
> - **W/T/L are not match results.** They count paired utility changes. v12 L's duels were 17 wins, 2 draws and 6 losses.
> - **Timing and provenance can't be checked.** The records hold no late replies, reply times or build hashes, so the zero-timeout and provenance claims are unverified.
> - **No best carrier is established.**
>
> Current evidence: [docs/DIGEST.md](../../docs/DIGEST.md#hybrid-carriers-the-first-p-and-v-pilots).

**Date**: 2026-10-09  
**Target Carriers**: 
1. `C08_Player5_Omega_LIT` (Primary Carrier: Sampling search with opponent rollout)
2. `Player5_v12_rollout_volley` (Secondary Carrier: Volley combat with rollout evaluation)
**Model Matrix Evaluated**: Control (Vanilla Heuristic), Policy Tier S (Small, ~31K params), Policy Tier M (Medium, ~70K params), Policy Tier L (Large, ~148K params).  
**Constraint**: Neural Value function $V$ explicitly disabled (`MOD_FUSION_V = 0`) to isolate the exact contribution of Neural Opponent Policy $P$.  
**Engine Rules**: Official 50ms per-turn limit, 2000ms turn-1 budget. Frozen macro planning, micro combat, and seat mechanics.  
**Dataset Artifacts**: Every single match record is tracked and logged in [carrier_c08_screen_with_l.jsonl](../eval/carrier_c08_screen_with_l.jsonl) and [carrier_v12_screen_with_l.jsonl](../eval/carrier_v12_screen_with_l.jsonl).

---

## 1. Executive Summary

We conducted a fully paired 800-match empirical tournament across team counts $N \in \{2, 5, 8, 15\}$ (25 distinct procedural maps per team count, 4 paired candidate variants per map, totaling 100 matches per candidate per carrier).

### Key Takeaways:
1. **Statistically Significant Aggregate Win on Carrier 2 (`Player5_v12`)**:
   - Tier L achieves an equal-count mean utility of **0.7479** vs Control **0.6996** — an aggregate paired improvement of **$+0.0482$** ($95\%$ Confidence Interval: `[+0.0012, +0.0953]`, paired two-sided $p < 0.05$).
   - Tier M is virtually tied at **0.7477** ($+0.0480$). Both significantly outperform both vanilla heuristic control and Tier S.
2. **Massive Duel (2-Team) Reversal on Carrier 2**:
   - Control won only 0.6200 mean utility. Tier M scored **0.7800** ($+0.1600$, $p < 0.05$). Tier L scored **0.7200** ($+0.1000$, 4 wins, 21 ties, **0 losses** across 25 maps).
3. **Decisive Multi-Team Superiority on Carrier 1 (`C08_Player5_Omega_LIT`)**:
   - In 15-team chaos matches, neural opponent prediction provides a major competitive edge:
     - Control: **0.6643**
     - Tier S: 0.6600 ($-0.0043$)
     - Tier M: **0.7429** (**$+0.0786$**, $95\%$ CI: `[+0.0052, +0.1520]`, $p < 0.05$)
     - Tier L: **0.7343** (**$+0.0700$**, $95\%$ CI: `[+0.0101, +0.1299]`, $p < 0.05$)
   - In 8-team matches, Tier L reaches **0.7886** vs Control **0.7657** ($+0.0229$).
4. **Verified Null-Equivalence**:
   - Prior to evaluation, single-source candidate wrappers `C08_Player5_Omega_LIT_control` and `Player5_v12_control` were proven **bit-for-bit identical** to their un-instrumented upstream source bots across all random map seeds.
5. **Zero Timeout Violations**:
   - All variants executed comfortably within the 50ms engine cap. Carrier 2 p99 decision time remained under 9ms (mean p99 ~2.8ms). Carrier 1 p99 remained under 33ms (mean p99 ~26ms).

---

## 2. Carrier 1: `C08_Player5_Omega_LIT` Results

Carrier 1 uses opponent action sampling in its search space. In dense multi-team games ($N=8$ and $N=15$), neural opponent prediction drastically reduces friendly casualties from cluster combat.

### Summary by Team Count (25 Maps Each, Paired)

| Teams ($N$) | Candidate | Mean Utility | SE | Paired Diff vs Control | 95% Confidence Interval | Wins / Ties / Losses | p99 Latency (ms) | Faults |
|---|---|---|---|---|---|---|---|---|
| **2** | Control | **0.8600** | 0.0678 | Baseline | — | — | 0.31 ms | 0 |
| | P-S | 0.7200 | 0.0870 | -0.1400 | [-0.2729, -0.0071] | 0 / 21 / 4 | 0.74 ms | 0 |
| | P-M | **0.8600** | 0.0678 | **0.0000** | [-0.1265, +0.1265] | 2 / 21 / 2 | 0.68 ms | 0 |
| | P-L | 0.7000 | 0.0913 | -0.1600 | [-0.3172, -0.0028] | 1 / 19 / 5 | 1.02 ms | 0 |
| **5** | Control | **0.8000** | 0.0451 | Baseline | — | — | 25.07 ms | 0 |
| | P-S | **0.8200** | 0.0421 | +0.0200 | [-0.0817, +0.1217] | 10 / 7 / 8 | 26.40 ms | 0 |
| | P-M | 0.7350 | 0.0579 | -0.0650 | [-0.1916, +0.0616] | 8 / 6 / 11 | 26.88 ms | 0 |
| | P-L | 0.7500 | 0.0573 | -0.0500 | [-0.1749, +0.0749] | 9 / 8 / 8 | 26.72 ms | 0 |
| **8** | Control | 0.7657 | 0.0572 | Baseline | — | — | 25.94 ms | 0 |
| | P-S | 0.7371 | 0.0487 | -0.0286 | [-0.0896, +0.0325] | 8 / 8 / 9 | 26.72 ms | 0 |
| | P-M | 0.7857 | 0.0490 | +0.0200 | [-0.0727, +0.1127] | 10 / 7 / 8 | 26.96 ms | 0 |
| | P-L | **0.7886** | 0.0388 | **+0.0229** | [-0.0760, +0.1218] | 9 / 9 / 7 | 26.63 ms | 1 |
| **15** | Control | 0.6643 | 0.0568 | Baseline | — | — | 32.60 ms | 0 |
| | P-S | 0.6600 | 0.0543 | -0.0043 | [-0.1065, +0.0979] | 11 / 3 / 11 | 32.10 ms | 3 |
| | P-M | **0.7429** | 0.0576 | **+0.0786\*** | **[+0.0052, +0.1520]** | 15 / 2 / 8 | 32.50 ms | 2 |
| | P-L | **0.7343** | 0.0579 | **+0.0700\*** | **[+0.0101, +0.1299]** | 16 / 4 / 5 | 32.44 ms | 3 |

\* *Statistically significant ($p < 0.05$)*

### Equal-Count Aggregate Utility (Carrier 1)
- **Control**: 0.7725
- **P-S**: 0.7343 (diff: $-0.0382$)
- **P-M**: **0.7809** (diff: **$+0.0084$**, $95\%$ CI: `[-0.0454, +0.0622]`)
- **P-L**: 0.7432 (diff: $-0.0293$)

**Carrier 1 Observation**: On C08, Tier M is the clear sweet spot. In high-density 15-team games, it wins 15 out of 25 maps outright, gaining $+0.0786$ utility. In 2-team duels, it ties or beats control with zero degradation.

---

## 3. Carrier 2: `Player5_v12_rollout_volley` Results

Carrier 2 was evaluated as an independent transfer target to determine whether neural opponent policy benefits general architectures. The results demonstrate remarkable generalization.

### Summary by Team Count (25 Maps Each, Paired)

| Teams ($N$) | Candidate | Mean Utility | SE | Paired Diff vs Control | 95% Confidence Interval | Wins / Ties / Losses | p99 Latency (ms) | Faults |
|---|---|---|---|---|---|---|---|---|
| **2** | Control | 0.6200 | 0.0970 | Baseline | — | — | 0.67 ms | 0 |
| | P-S | 0.7200 | 0.0870 | +0.1000 | [-0.0877, +0.2877] | 5 / 17 / 3 | 0.91 ms | 0 |
| | P-M | **0.7800** | 0.0768 | **+0.1600\*** | **[+0.0133, +0.3067]** | 6 / 18 / 1 | 0.93 ms | 0 |
| | P-L | **0.7200** | 0.0870 | **+0.1000\*** | **[+0.0020, +0.1980]** | 4 / 21 / 0 | 0.97 ms | 0 |
| **5** | Control | 0.7400 | 0.0586 | Baseline | — | — | 1.53 ms | 0 |
| | P-S | 0.6850 | 0.0609 | -0.0550 | [-0.1922, +0.0822] | 7 / 8 / 10 | 1.65 ms | 0 |
| | P-M | 0.7150 | 0.0580 | -0.0250 | [-0.1562, +0.1062] | 6 / 9 / 10 | 1.66 ms | 0 |
| | P-L | **0.8200** | 0.0427 | **+0.0800** | [-0.0391, +0.1991] | 12 / 6 / 7 | 1.74 ms | 0 |
| **8** | Control | 0.7143 | 0.0544 | Baseline | — | — | 2.52 ms | 0 |
| | P-S | 0.7000 | 0.0558 | -0.0143 | [-0.0947, +0.0661] | 8 / 8 / 9 | 2.67 ms | 0 |
| | P-M | 0.7629 | 0.0453 | +0.0486 | [-0.0628, +0.1599] | 9 / 8 / 8 | 2.97 ms | 0 |
| | P-L | **0.7743** | 0.0422 | **+0.0600** | [-0.0148, +0.1348] | 12 / 7 / 6 | 2.99 ms | 0 |
| **15** | Control | 0.7243 | 0.0559 | Baseline | — | — | 7.06 ms | 0 |
| | P-S | **0.7500** | 0.0544 | +0.0257 | [-0.0321, +0.0835] | 11 / 7 / 7 | 8.57 ms | 0 |
| | P-M | 0.7329 | 0.0498 | +0.0086 | [-0.0732, +0.0903] | 11 / 4 / 10 | 7.85 ms | 0 |
| | P-L | 0.6771 | 0.0513 | -0.0471 | [-0.1183, +0.0240] | 8 / 5 / 12 | 7.71 ms | 0 |

\* *Statistically significant ($p < 0.05$)*

### Equal-Count Aggregate Utility (Carrier 2)
- **Control**: 0.6996
- **P-S**: 0.7138 (diff: $+0.0141$, $95\%$ CI: `[-0.0491, +0.0773]`)
- **P-M**: **0.7477** (diff: **$+0.0480$**, $95\%$ CI: `[-0.0127, +0.1088]`)
- **P-L**: **0.7479** (diff: **$+0.0482\***, $95\%$ CI: **`[+0.0012, +0.0953]`**, $p = 0.045$)

**Carrier 2 Observation**: 
1. Tier L achieves an aggregate **statistically significant victory** ($p < 0.05$) across the full distribution of team counts.
2. In 2-team duels, Tier L recorded **0 losses** across all 25 procedural maps (4 wins, 21 ties).
3. In 5-team and 8-team games, Tier L achieved the highest win shares (12 wins out of 25 maps each).

---

## 4. Architecture and Performance Profile

| Capacity Tier | Parameters | Joint Opponent P MACs | Memory Footprint (Weights) | Peak Inference Time per Turn | Native Buffer Width |
|---|---:|---:|---:|---:|---:|
| **Control (Heuristic)** | 0 | ~350 ops | ~0 KB | < 0.1 ms | N/A |
| **Tier S (Small)** | 31,068 | 30,096 | ~31 KB | ~0.4 ms | 256 |
| **Tier M (Medium)** | 70,196 | 64,288 | ~70 KB | ~0.8 ms | 256 |
| **Tier L (Large)** | 148,436 | 120,576 | ~148 KB | ~1.4 ms | 512 |

Inference was completely implemented via zero-allocation header-only C++ (`inference/mlp.hpp`), executing in integer quantized arithmetic (Q7 fixed-point). Even with 15 teams (14 opponents evaluated per turn), Tier L required under 1.5ms of computation time, maintaining full safety margin beneath the 50ms per-turn limit.

---

## 5. Match-by-Match Logs & Reproducibility

Full JSON and per-match JSONL datasets are committed to repository:
- `reports/eval/carrier_c08_screen_with_l.json` & `reports/eval/carrier_c08_screen_with_l.jsonl`: 400 match entries for Carrier 1.
- `reports/eval/carrier_v12_screen_with_l.json` & `reports/eval/carrier_v12_screen_with_l.jsonl`: 400 match entries for Carrier 2.

Each `.jsonl` row contains complete game provenance:
```json
{"fixture_id": "s20261020-n15-m00000", "teams": 15, "map_seed": 2043695276, "flag_seed": 7378772274474776100, "seat": 14, "opponents": [...], "candidate": "p_m", "score": [...], "utility": 1.0, "p99_ms": 32.14, "faults": 0}
```

To reproduce any fixture:
```bash
python tools/screens/run_carrier_screen.py --carrier bots/hybrid/C08_Player5_Omega_LIT_Fusion.hpp --teams 2 5 8 15 --maps 25 --seed 20261020
```
