# Refreshed Per-Count Opponent P Specialists (T0067) Report

**Task ID:** `T0067`  
**Feature Family:** `p2` (923 features)  
**Capacity:** `XXL` (`mover: [256]`, `squad: [1024, 256]`, `act: [256]`, `target: [32]`)  
**Optimizer Seeds:** `101` (primary deployment), `102`, `103` (replication diagnostics)  
**Training Epochs:** 8 epochs per seed with validation checkpoint selection  
**Learning Rate Schedule:** Constant LR 0.001 with Adam (docs/MEOWMIND_RL_PLAN.md baseline; cosine decay omitted)  
**Pure P Deployment Alpha:** $\alpha = 1.0$ (no blending with legacy $q_H$ heuristic)  
**Dispatch & Routing:** [`inference/count_specialists_p/registry.hpp`](../../../inference/count_specialists_p/registry.hpp) selects exact count in $O(1)$ from `INIT` without falling back or running all 14 specialists per decision.  

---

## 1. Executive Summary

All 14 count-specific opponent P specialists (exact team counts 2 through 15) were trained on their dedicated, map-disjoint `nXX` corpora produced by T0066.
Each specialist was warm-started from the corresponding deployed XXL baseline (or `weights_general.npz` for previously missing counts `n09`, `n11`, `n13`, `n14`), trained across three optimizer seeds, calibrated on validation data, evaluated on strictly held-out test data in both float and int16 formats, quantized to int16 headers, and verified for exact bitwise C++ native parity.

---

## 2. Quantized Int16 Test Evaluation vs Existing Baseline (with Map-Cluster Intervals)

Each exported int16 model (`weights_nXX.npz`) was evaluated against the deployed baseline on held-out test matches (30 independent maps per count).
Standard error bounds and 95% confidence intervals are clustered by independent map seed (`training.evaluate_value.clustered_mean_ci`):

| Count | Specialist | Warm-Start Source | Baseline Int16 Test NLL | Specialist Int16 Test NLL | Δ Loss (Gain) | 95% Map-Cluster CI | Verdict | Top-1 Acc | Top-5 Acc | Act Acc | Target Acc |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | `n02` | `weights_n02.npz` | 1.8004 | **1.6279** | **+0.1725** | `[-0.2070, -0.1381]` | **lower** | 52.36% | 85.51% | 67.94% | 20.15% |
| 3 | `n03` | `weights_n03.npz` | 2.0683 | **1.9489** | **+0.1194** | `[-0.1533, -0.0854]` | **lower** | 45.92% | 80.32% | 77.03% | 13.94% |
| 4 | `n04` | `weights_n04.npz` | 2.0518 | **1.9668** | **+0.0850** | `[-0.1158, -0.0542]` | **lower** | 46.39% | 79.99% | 74.17% | 14.81% |
| 5 | `n05` | `weights_n05.npz` | 2.2794 | **2.1960** | **+0.0834** | `[-0.1012, -0.0655]` | **lower** | 39.23% | 75.51% | 67.63% | 18.72% |
| 6 | `n06` | `weights_n06.npz` | 2.1502 | **2.0824** | **+0.0678** | `[-0.0817, -0.0539]` | **lower** | 43.00% | 77.48% | 75.54% | 14.89% |
| 7 | `n07` | `weights_n07.npz` | 2.1643 | **2.1001** | **+0.0642** | `[-0.0746, -0.0537]` | **lower** | 42.43% | 77.38% | 73.88% | 16.92% |
| 8 | `n08` | `weights_n08.npz` | 2.1962 | **2.1387** | **+0.0575** | `[-0.0702, -0.0448]` | **lower** | 41.81% | 76.56% | 71.68% | 17.90% |
| 9 | `n09` | `weights_general.npz` | 2.1182 | **2.0603** | **+0.0579** | `[-0.0656, -0.0503]` | **lower** | 42.60% | 78.45% | 68.47% | 19.53% |
| 10 | `n10` | `weights_n10.npz` | 2.1498 | **2.0921** | **+0.0577** | `[-0.0657, -0.0496]` | **lower** | 42.14% | 77.72% | 66.81% | 19.92% |
| 11 | `n11` | `weights_general.npz` | 2.0685 | **2.0251** | **+0.0434** | `[-0.0482, -0.0386]` | **lower** | 44.11% | 79.19% | 71.49% | 17.92% |
| 12 | `n12` | `weights_n12.npz` | 2.1025 | **2.0463** | **+0.0562** | `[-0.0613, -0.0511]` | **lower** | 43.44% | 78.78% | 70.17% | 19.03% |
| 13 | `n13` | `weights_general.npz` | 2.0741 | **2.0274** | **+0.0467** | `[-0.0521, -0.0414]` | **lower** | 43.82% | 79.31% | 69.16% | 19.53% |
| 14 | `n14` | `weights_general.npz` | 2.0637 | **2.0212** | **+0.0425** | `[-0.0474, -0.0375]` | **lower** | 44.10% | 79.40% | 67.52% | 19.96% |
| 15 | `n15` | `weights_n15.npz` | 2.0717 | **2.0168** | **+0.0549** | `[-0.0602, -0.0496]` | **lower** | 43.91% | 79.71% | 66.74% | 20.66% |

---

## 3. Float Model Training & Calibration Summary

| Count | Specialist | Warm-Start Source | Baseline Float Val NLL | Specialist Float Val NLL | Selected Epoch | Calibrated Beta (J / A / T) |
|:---:|:---:|---|:---:|:---:|:---:|:---:|
| 2 | `n02` | `weights_n02.npz` | 1.7089 | **1.5501** | Epoch 7 | 1.03 / 1.01 / 1.03 |
| 3 | `n03` | `weights_n03.npz` | 2.0475 | **1.9840** | Epoch 3 | 1.03 / 1.00 / 1.01 |
| 4 | `n04` | `weights_n04.npz` | 2.1278 | **2.0209** | Epoch 3 | 1.02 / 1.02 / 0.98 |
| 5 | `n05` | `weights_n05.npz` | 2.2110 | **2.1483** | Epoch 1 | 1.03 / 0.96 / 1.04 |
| 6 | `n06` | `weights_n06.npz` | 2.1426 | **2.0550** | Epoch 3 | 1.00 / 1.04 / 1.06 |
| 7 | `n07` | `weights_n07.npz` | 2.1808 | **2.1124** | Epoch 1 | 1.05 / 1.04 / 1.05 |
| 8 | `n08` | `weights_n08.npz` | 2.1544 | **2.0877** | Epoch 2 | 1.05 / 1.02 / 0.98 |
| 9 | `n09` | `weights_general.npz` | 2.1242 | **2.0729** | Epoch 1 | 1.04 / 1.01 / 1.05 |
| 10 | `n10` | `weights_n10.npz` | 2.1663 | **2.1071** | Epoch 1 | 1.05 / 1.02 / 1.02 |
| 11 | `n11` | `weights_general.npz` | 2.0934 | **2.0466** | Epoch 1 | 1.04 / 0.99 / 1.00 |
| 12 | `n12` | `weights_n12.npz` | 2.0934 | **2.0345** | Epoch 1 | 1.02 / 1.05 / 1.05 |
| 13 | `n13` | `weights_general.npz` | 2.0700 | **2.0211** | Epoch 1 | 1.02 / 1.02 / 1.02 |
| 14 | `n14` | `weights_general.npz` | 2.0943 | **2.0484** | Epoch 1 | 1.07 / 0.97 / 1.00 |
| 15 | `n15` | `weights_n15.npz` | 2.0882 | **2.0355** | Epoch 1 | 1.05 / 1.01 / 1.02 |

---

## 4. Seed Replication Diagnostics (Seeds 101, 102, 103)

Validation total loss across three independent optimizer seeds demonstrates consistent convergence without variance collapse:

| Count | Model | Seed 101 Best Val (Primary) | Seed 102 Best Val | Seed 103 Best Val | Max Spread (Δ Loss) | Selected Epoch |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | `n02` | 1.7283 | 1.7295 | 1.7301 | 0.0018 | Epoch 7 |
| 3 | `n03` | 2.1610 | 2.1565 | 2.1578 | 0.0045 | Epoch 3 |
| 4 | `n04` | 2.2217 | 2.2247 | 2.2227 | 0.0030 | Epoch 3 |
| 5 | `n05` | 2.3870 | 2.3840 | 2.3842 | 0.0030 | Epoch 1 |
| 6 | `n06` | 2.2254 | 2.2208 | 2.2206 | 0.0048 | Epoch 3 |
| 7 | `n07` | 2.2974 | 2.2977 | 2.2952 | 0.0025 | Epoch 1 |
| 8 | `n08` | 2.2856 | 2.2856 | 2.2856 | 0.0000 | Epoch 2 |
| 9 | `n09` | 2.2809 | 2.2817 | 2.2779 | 0.0038 | Epoch 1 |
| 10 | `n10` | 2.3291 | 2.3260 | 2.3295 | 0.0035 | Epoch 1 |
| 11 | `n11` | 2.2486 | 2.2472 | 2.2505 | 0.0033 | Epoch 1 |
| 12 | `n12` | 2.2352 | 2.2314 | 2.2310 | 0.0042 | Epoch 1 |
| 13 | `n13` | 2.2295 | 2.2305 | 2.2284 | 0.0021 | Epoch 1 |
| 14 | `n14` | 2.2734 | 2.2697 | 2.2701 | 0.0037 | Epoch 1 |
| 15 | `n15` | 2.2592 | 2.2578 | 2.2579 | 0.0014 | Epoch 1 |

---

## 5. Native C++ Export & Parity Verification

All 14 exported headers in `inference/count_specialists_p/` were compiled via `native.infer_check` and evaluated against Python `QuantizedTier.raw` on raw game streams.

| Model | Exported Header | Header Size | NPZ Size | Namespace | Include Guard | Parity Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| `n02` | `weights_n02.hpp` | 4.21 MB | 0.78 MB | `fusion::models::csp_n02` | `FUSION_MODEL_CSP_N02_HPP` | **PASSED (0 diff)** |
| `n03` | `weights_n03.hpp` | 4.21 MB | 0.78 MB | `fusion::models::csp_n03` | `FUSION_MODEL_CSP_N03_HPP` | **PASSED (0 diff)** |
| `n04` | `weights_n04.hpp` | 4.20 MB | 0.78 MB | `fusion::models::csp_n04` | `FUSION_MODEL_CSP_N04_HPP` | **PASSED (0 diff)** |
| `n05` | `weights_n05.hpp` | 4.20 MB | 0.77 MB | `fusion::models::csp_n05` | `FUSION_MODEL_CSP_N05_HPP` | **PASSED (0 diff)** |
| `n06` | `weights_n06.hpp` | 4.15 MB | 0.76 MB | `fusion::models::csp_n06` | `FUSION_MODEL_CSP_N06_HPP` | **PASSED (0 diff)** |
| `n07` | `weights_n07.hpp` | 4.17 MB | 0.76 MB | `fusion::models::csp_n07` | `FUSION_MODEL_CSP_N07_HPP` | **PASSED (0 diff)** |
| `n08` | `weights_n08.hpp` | 4.11 MB | 0.75 MB | `fusion::models::csp_n08` | `FUSION_MODEL_CSP_N08_HPP` | **PASSED (0 diff)** |
| `n09` | `weights_n09.hpp` | 4.12 MB | 0.76 MB | `fusion::models::csp_n09` | `FUSION_MODEL_CSP_N09_HPP` | **PASSED (0 diff)** |
| `n10` | `weights_n10.hpp` | 4.10 MB | 0.73 MB | `fusion::models::csp_n10` | `FUSION_MODEL_CSP_N10_HPP` | **PASSED (0 diff)** |
| `n11` | `weights_n11.hpp` | 4.09 MB | 0.74 MB | `fusion::models::csp_n11` | `FUSION_MODEL_CSP_N11_HPP` | **PASSED (0 diff)** |
| `n12` | `weights_n12.hpp` | 4.05 MB | 0.71 MB | `fusion::models::csp_n12` | `FUSION_MODEL_CSP_N12_HPP` | **PASSED (0 diff)** |
| `n13` | `weights_n13.hpp` | 4.06 MB | 0.73 MB | `fusion::models::csp_n13` | `FUSION_MODEL_CSP_N13_HPP` | **PASSED (0 diff)** |
| `n14` | `weights_n14.hpp` | 4.06 MB | 0.73 MB | `fusion::models::csp_n14` | `FUSION_MODEL_CSP_N14_HPP` | **PASSED (0 diff)** |
| `n15` | `weights_n15.hpp` | 4.00 MB | 0.68 MB | `fusion::models::csp_n15` | `FUSION_MODEL_CSP_N15_HPP` | **PASSED (0 diff)** |

---

## 6. Host Feasibility Receipt (8-Worker Concurrent Host Benchmark)

- **Benchmark Kind:** `host 15-team eight-worker concurrent benchmark under lock cpu`
- **Concurrent Workers:** **8** under task CPU lease.
- **Teams Evaluated:** **15** teams.
- **Initialization Latency (15 teams):** `28.74 ms` (limit: `2,000 ms`). **PASSED**.
- **Per-Turn Inference Latency (Mean):** `1.191 ms` (limit: `50 ms`). **PASSED**.
- **Per-Turn Inference Latency (P99):** `1.217 ms`.
- **Per-Turn Inference Latency (Max):** `1.217 ms`.
- **Per-Decision Specialist Calls:** Exactly **1** model evaluated per turn via O(1) table lookup in `registry.hpp`. The bot never runs all 14 models per decision.

---

## 7. Artifact Inventory

- **Inference Directory:** [`inference/count_specialists_p/`](../../../inference/count_specialists_p/)
  - 14 quantized C++ headers: `weights_n02.hpp` .. `weights_n15.hpp` (namespaced `csp_n02` .. `csp_n15`)
  - 14 quantized npz arrays: `weights_n02.npz` .. `weights_n15.npz`
  - Dispatch table: `registry.hpp` (`fusion::models::count_specialists_p::forTeams`)
  - Complete artifact manifest: `MANIFEST.json`
- **Training Receipts & Diagnostics:** [`reports/models/count_specialists_p/`](../../../reports/models/count_specialists_p/)
- **System Profile:** [`profiles/count_specialists_p.json`](../../../profiles/count_specialists_p.json)
