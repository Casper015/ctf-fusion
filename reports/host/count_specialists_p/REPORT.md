# Refreshed Per-Count Opponent P Specialists (T0067) Report

**Task ID:** `T0067`  
**Feature Family:** `p2` (923 features)  
**Capacity:** `XXL` (`mover: [256]`, `squad: [1024, 256]`, `act: [256]`, `target: [32]`)  
**Optimizer Seeds:** `101` (primary deployment), `102`, `103` (replication diagnostics)  
**Training Epochs:** 8 epochs per seed with validation checkpoint selection  
**Dispatch & Routing:** [`inference/count_specialists_p/registry.hpp`](../../../inference/count_specialists_p/registry.hpp) selects exact count in $O(1)$ from `INIT` without falling back or running all 14 specialists per decision.  

---

## 1. Executive Summary

All 14 count-specific opponent P specialists (exact team counts 2 through 15) were trained on their dedicated, map-disjoint `nXX` corpora produced by T0066.
Each specialist was warm-started from the corresponding deployed XXL baseline (or `weights_general.npz` for previously missing counts `n09`, `n11`, `n13`, `n14`), trained across three optimizer seeds, calibrated on validation data, evaluated on strictly held-out test data, quantized to int16, and verified for exact bitwise C++ native parity.

---

## 2. Held-Out Test Evaluation vs Existing Baseline

| Count | Specialist | Warm-Start Source | Baseline Test NLL | Specialist Test NLL | Joint Top-1 Acc | Joint Top-5 Acc | Act Acc | Target Acc | Calibrated Beta (J / A / T) |
|:---:|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | `n02` | `weights_n02.npz` | 1.8003 | **1.6272** | 52.42% | 85.48% | 67.93% | 20.15% | 1.03 / 1.01 / 1.03 |
| 3 | `n03` | `weights_n03.npz` | 2.0678 | **1.9480** | 45.93% | 80.34% | 77.04% | 13.94% | 1.03 / 1.00 / 1.01 |
| 4 | `n04` | `weights_n04.npz` | 2.0517 | **1.9661** | 46.37% | 80.00% | 74.18% | 14.81% | 1.02 / 1.02 / 0.98 |
| 5 | `n05` | `weights_n05.npz` | 2.2792 | **2.1960** | 39.20% | 75.48% | 67.64% | 18.72% | 1.03 / 0.96 / 1.04 |
| 6 | `n06` | `weights_n06.npz` | 2.1500 | **2.0816** | 42.98% | 77.50% | 75.53% | 14.91% | 1.00 / 1.04 / 1.06 |
| 7 | `n07` | `weights_n07.npz` | 2.1641 | **2.0996** | 42.43% | 77.44% | 73.88% | 16.92% | 1.05 / 1.04 / 1.05 |
| 8 | `n08` | `weights_n08.npz` | 2.1959 | **2.1381** | 41.81% | 76.58% | 71.68% | 17.90% | 1.05 / 1.02 / 0.98 |
| 9 | `n09` | `weights_general.npz` | 2.1181 | **2.0598** | 42.61% | 78.45% | 68.47% | 19.53% | 1.04 / 1.01 / 1.05 |
| 10 | `n10` | `weights_n10.npz` | 2.1496 | **2.0915** | 42.15% | 77.70% | 66.82% | 19.91% | 1.05 / 1.02 / 1.02 |
| 11 | `n11` | `weights_general.npz` | 2.0683 | **2.0245** | 44.13% | 79.20% | 71.49% | 17.93% | 1.04 / 0.99 / 1.00 |
| 12 | `n12` | `weights_n12.npz` | 2.1022 | **2.0461** | 43.46% | 78.79% | 70.17% | 19.03% | 1.02 / 1.05 / 1.05 |
| 13 | `n13` | `weights_general.npz` | 2.0739 | **2.0269** | 43.80% | 79.32% | 69.17% | 19.53% | 1.02 / 1.02 / 1.02 |
| 14 | `n14` | `weights_general.npz` | 2.0634 | **2.0207** | 44.10% | 79.37% | 67.53% | 19.96% | 1.07 / 0.97 / 1.00 |
| 15 | `n15` | `weights_n15.npz` | 2.0714 | **2.0163** | 43.94% | 79.74% | 66.74% | 20.66% | 1.05 / 1.01 / 1.02 |

---

## 3. Seed Replication Diagnostics (Seeds 101, 102, 103)

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

## 4. Native C++ Export & Parity Verification

All 14 exported headers in `inference/count_specialists_p/` were compiled via `native.infer_check` and evaluated against Python `QuantizedTier.raw` on raw game streams.

| Model | Exported Header | Header Size | NPZ Size | Rows Compared | Max Abs Difference | Mismatch Count | Parity Status |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `n02` | `weights_n02.hpp` | 4.25 MB | 0.78 MB | 800 | **0** | **0** | **PASSED** |
| `n03` | `weights_n03.hpp` | 4.25 MB | 0.78 MB | 1,200 | **0** | **0** | **PASSED** |
| `n04` | `weights_n04.hpp` | 4.24 MB | 0.78 MB | 1,600 | **0** | **0** | **PASSED** |
| `n05` | `weights_n05.hpp` | 4.24 MB | 0.77 MB | 2,000 | **0** | **0** | **PASSED** |
| `n06` | `weights_n06.hpp` | 4.19 MB | 0.76 MB | 2,400 | **0** | **0** | **PASSED** |
| `n07` | `weights_n07.hpp` | 4.21 MB | 0.76 MB | 2,800 | **0** | **0** | **PASSED** |
| `n08` | `weights_n08.hpp` | 4.15 MB | 0.75 MB | 3,200 | **0** | **0** | **PASSED** |
| `n09` | `weights_n09.hpp` | 4.16 MB | 0.76 MB | 3,600 | **0** | **0** | **PASSED** |
| `n10` | `weights_n10.hpp` | 4.14 MB | 0.73 MB | 4,000 | **0** | **0** | **PASSED** |
| `n11` | `weights_n11.hpp` | 4.13 MB | 0.74 MB | 4,400 | **0** | **0** | **PASSED** |
| `n12` | `weights_n12.hpp` | 4.09 MB | 0.71 MB | 4,800 | **0** | **0** | **PASSED** |
| `n13` | `weights_n13.hpp` | 4.10 MB | 0.73 MB | 5,200 | **0** | **0** | **PASSED** |
| `n14` | `weights_n14.hpp` | 4.10 MB | 0.73 MB | 5,600 | **0** | **0** | **PASSED** |
| `n15` | `weights_n15.hpp` | 4.04 MB | 0.68 MB | 6,000 | **0** | **0** | **PASSED** |

---

## 5. Host Feasibility Receipt

- **Initialization Latency (15 teams):** `35.39 ms` (official limit: `2,000 ms`). **PASSED**.
- **Per-Turn Inference Latency (15 teams):** `0.838 ms` (official limit: `50 ms`). **PASSED**.
- **Per-Decision Specialist Calls:** Exactly **1** model evaluated per turn via O(1) table lookup in `registry.hpp`. The bot never runs all 14 models per decision.

---

## 6. Artifact Inventory

- **Inference Directory:** [`inference/count_specialists_p/`](../../../inference/count_specialists_p/)
  - 14 quantized C++ headers: `weights_n02.hpp` .. `weights_n15.hpp`
  - 14 quantized npz arrays: `weights_n02.npz` .. `weights_n15.npz`
  - Dispatch table: `registry.hpp` (`fusion::models::count_specialists_p::forTeams`)
  - Complete artifact manifest: `MANIFEST.json`
- **Training Receipts & Diagnostics:** [`reports/models/count_specialists_p/`](../../../reports/models/count_specialists_p/)
- **System Profile:** [`profiles/count_specialists_p.json`](../../../profiles/count_specialists_p.json)
