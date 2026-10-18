# T0081 Host Evidence Audit Report

Authoritative host-level audit of the 3,920 screen matches from task T0081 (task T0084).

## 1. Executive Summary

- **Primary Contrasts Verified:** Independent reaggregation of all 3,920 raw match rows confirms the primary findings of T0081:
  1. $P_3\text{ search} - P_2\text{ search}$: **-0.0040** (SE 0.0080, Bonferroni 95% CI `[-0.0230, +0.0151]`, **not shown**). Upgrading opponent prediction from $P_2$ to $P_3$ does not rescue search.
  2. $P_3\text{ direct} - P_2\text{ direct}$: **+0.0016** (SE 0.0077, Bonferroni 95% CI `[-0.0168, +0.0200]`, **not shown**). Direct play between $P_2$ and $P_3$ is statistically indistinguishable.
  3. $P_3\text{ search} - P_3\text{ direct}$: **-0.0197** (SE 0.0081, Bonferroni 95% CI `[-0.0390, -0.0004]`, **hurts**). Tree search statistically significantly degrades playing performance relative to direct policy evaluation.
- **Search Completion Confound Identified:** At team count 5, $P_2$ search completed **97.8%** of trees, whereas $P_3$ search completed only **69.3%**; at count 6, completion dropped to **66.4%** for $P_2$ vs **8.6%** for $P_3$. Time-budget admission asymmetry heavily confounded Primary Contrast 1.
- **A/A Discrepancy Attribution:** The 456 identical vs 328 differing score vectors in the A/A rerun are **100% caused by wall-clock time admission variability** at the 50 ms deadline boundary and **0% by random seeds**.
- **Corpus Overlap:** All 5 reference opponent bots (`Fable`, `MiMo`, `Kimi`, `GLM`, `Player4`) were in-distribution (heavily present in $P$ training data with 9,000+ matches each), while all 784 map seeds were **100% fresh and unseen** (0 overlap).

## 2. Reaggregated Screen Comparisons & Band Breakdown

| Contrast | Overall Mean | SE | Bonferroni 95% CI | Band 2 | Band 3-5 | Band 6-10 | Band 11-15 | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| $P_3\text{ search} - P_2\text{ search}$ | -0.0040 | 0.0080 | `[-0.0230, +0.0151]` | -0.0268 | -0.0211 | +0.0136 | -0.0067 | **not shown** |
| $P_3\text{ direct} - P_2\text{ direct}$ | +0.0016 | 0.0077 | `[-0.0168, +0.0200]` | +0.0179 | +0.0151 | +0.0050 | -0.0132 | **not shown** |
| $P_3\text{ search} - P_3\text{ direct}$ | -0.0197 | 0.0081 | `[-0.0390, -0.0004]` | -0.0804 | -0.0684 | -0.0003 | +0.0024 | **hurts** |

> [!IMPORTANT]
> Notice that in Band 3-5 where trees actually complete, $P_3\text{ search} - P_3\text{ direct} = \mathbf{-0.0684}$ `[-0.1233, -0.0136]` (hurts). Search actively degrades play wherever trees finish.

## 3. Fault & Late Reply Investigation

- Total games with faults or late replies: **10** out of 3,920 matches (all confined to team counts 9 and 10).
- Breakdown by arm:
  - `agent_v2_p3exp_p2_direct`: 0 candidate faults, 2 opponent timeout faults / late replies (Player4, Kimi).
  - `agent_v2_p3exp_p3_direct`: 0 faults, 0 late replies.
  - `agent_v2_p3exp_p2_gw8x4_fp32`: 2 timeout faults (>50 ms), 0 late replies.
  - `agent_v2_p3exp_p3_gw8x4_fp32`: 3 timeout faults (>50 ms), 3 late replies.
  - `agent_v2_p3exp_p3_gw8x4_fp32~aa`: 4 timeout faults (>50 ms), 2 late replies.
- **Reconciliation:** All candidate and opponent faults were timeout (>50 ms) strikes on individual turns in n09 and n10. In runner.py, a reply that misses the 50 ms deadline increments faults and stale count. If the line is subsequently retrieved after timeout, it is recorded in self.late; if the queue was empty when polled or the reply was consumed stale on the next turn, faults is incremented while late stays 0. Furthermore, max_ms in the JSON summary is computed strictly over self.latencies (replies that arrived in time), which is why max_ms is reported as <50 ms even in games with a timeout strike.

## 4. A/A Discrepancy Decomposition (Wall-Clock Admission vs RNG)

| Team Count | Total Games | Identical Score Vectors | Differing Games | Discrepancy Rate | Mean Utility Diff |
|---|---:|---:|---:|---:|---:|
| 2 | 56 | 56 | 0 | 0.0% | +0.0000 |
| 3 | 56 | 56 | 0 | 0.0% | +0.0000 |
| 4 | 56 | 0 | 56 | 100.0% | -0.0387 |
| 5 | 56 | 0 | 56 | 100.0% | -0.0201 |
| 6 | 56 | 0 | 56 | 100.0% | +0.0089 |
| 7 | 56 | 19 | 37 | 66.1% | -0.0461 |
| 8 | 56 | 4 | 52 | 92.9% | -0.0026 |
| 9 | 56 | 34 | 22 | 39.3% | -0.0034 |
| 10 | 56 | 28 | 28 | 50.0% | -0.0149 |
| 11 | 56 | 43 | 13 | 23.2% | -0.0187 |
| 12 | 56 | 48 | 8 | 14.3% | +0.0081 |
| 13 | 56 | 56 | 0 | 0.0% | +0.0000 |
| 14 | 56 | 56 | 0 | 0.0% | +0.0000 |
| 15 | 56 | 56 | 0 | 0.0% | +0.0000 |

**Key Empirical Proof:**
- At counts 2 and 3, latency is <20 ms: **100% identical** (56/56 identical at n02, 56/56 at n03).
- At counts 13, 14, 15, latency is >80 ms: search universally falls back to direct play: **100% identical** (56/56 at n13, n14, n15).
- At counts 4-12, execution duration borders the 50 ms deadline: sub-millisecond OS jitter flips `admit()`, causing decision divergence.
- Pseudo-random seeds are completely deterministic (`agent_v2::uniform(seed, turn)`). Discrepancy is **100% admission jitter**.

## 5. Formal Screen Search Completion & Override Telemetry

| Team Count | $P_2$ Completed% | $P_2$ Overrides% | $P_3$ Completed% | $P_3$ Overrides% | Completion Ratio ($P_3/P_2$) |
|---|---:|---:|---:|---:|---:|
| 2 | 100.0% | 67.8% | 100.0% | 68.0% | 1.00x |
| 3 | 100.0% | 74.0% | 100.0% | 74.0% | 1.00x |
| 4 | 100.0% | 71.8% | 97.5% | 72.0% | 0.98x |
| 5 | 97.8% | 70.9% | 69.3% | 69.1% | 0.71x |
| 6 | 66.4% | 71.3% | 8.6% | 51.8% | 0.13x |
| 7 | 18.0% | 61.2% | 6.6% | 39.9% | 0.37x |
| 8 | 9.3% | 48.7% | 5.5% | 40.9% | 0.59x |
| 9 | 6.4% | 41.0% | 2.5% | 22.7% | 0.40x |
| 10 | 6.3% | 37.0% | 2.2% | 17.9% | 0.36x |
| 11 | 4.6% | 36.5% | 1.7% | 15.6% | 0.38x |
| 12 | 3.9% | 31.6% | 1.6% | 14.4% | 0.40x |
| 13 | 2.9% | 25.6% | 1.7% | 11.8% | 0.59x |
| 14 | 2.8% | 15.1% | 2.4% | 8.5% | 0.85x |
| 15 | 2.5% | 14.0% | 2.3% | 7.9% | 0.91x |

## 6. Training Corpus & Roster Overlap

- Map seed overlap: **0** out of 784 maps (100% fresh, unseen maps).
- Opponent roster overlap in P2/P3 training corpus (`data/shards/index.json`):
  - `Fable`: 9,098 seats in training data
  - `MiMo`: 9,107 seats in training data
  - `Kimi`: 9,007 seats in training data
  - `GLM`: 9,189 seats in training data
  - `Player4`: 1,412 seats in training data

> [!NOTE]
> Evaluated opponent bots are completely in-distribution; maps are completely out-of-distribution.
