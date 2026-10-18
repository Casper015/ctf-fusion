# Search Failure Decomposition & Diagnostic Report

**Diagnosis Target:** Explaining the gap between Meowmind (+8.09 pp) and agent_v2 search (-3.10 pp).

## Executive Findings

| Dimension | Audit Focus | Metric / Finding | Impact on Search Gap |
|---|---|---|---|
| **1. 候选生成 (Candidates)** | Shortlist headroom | Top-8 contains +0.042 utility headroom over direct Top-1 | **Not the bottleneck** (good candidates exist) |
| **2. 对手预测 (P Fidelity)** | P2 vs P3 vs Ground Truth | P3 improves NLL by -0.063 nats; P2 error in combat simulates unreal branches | **Secondary driver** (T0081 tests if P3 fixes this) |
| **3. 模拟器保真度 (Simulator)** | Parity vs official engine | HP, collision, flag states match official engine rules | **Not the cause** (mechanics are faithful) |
| **4. 价值排序与反悔 (V Ranking)** | Correlation & Regret | Overrides hurt in 60.8% of cases (mean regret -0.0168) | **PRIMARY ROOT CAUSE** (V picks wrong candidates) |
| **5. 算力与时延分配 (Compute)** | Tree completion by count | n15 times out 65.9% of turns; n02-n05 search completes but hurts | **High-count bottleneck** |

## Strategic Recommendations

1. **T0081 (P3 vs P2):** Directly tests whether Dimension 2 (better opponent prediction) mitigates search loss.
2. **If P3 does not fix search:** Collect search-branch decision data (Dimension 4) to train candidate-relative advantage / ranker rather than expanding unguided V.
