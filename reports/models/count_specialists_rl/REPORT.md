# Exact-Count PPO Policy and Value Specialists (T0068) Report

**Task ID:** `T0068`  
**Carrier:** Modular `agent_v2_v1` carrier (task T0065)  
**Opponent Predictor Bank:** Frozen refreshed `count_specialists_p` (`FUSION_TIER_CSP`, 2048 buffer width, task T0067)  
**Opponent Pool:** 50 diverse training pool bots (`profiles/meow_rl_training_pool_v2.json`)  
**Actor Architecture:** 112 -> 64 -> 32 -> 1 (`fusion::Net`, reference16 INT16)  
**Critic Architecture:** Large V 416 -> 1024 -> 1024 -> 512 -> 1  
**Total Training Budget:** **2,800,000 learner decisions** (1.4M specialized arm + 1.4M general control arm)  
**Host Platform:** `desktop-g60ojud` (RTX 4080 Laptop GPU, 8 CPU rollout workers)  

---

## 1. Executive Summary

Task T0068 executes the reinforcement learning phase of the count-specialist program.
Using the unified modular `agent_v2` carrier and frozen refreshed per-count opponent predictor bank `count_specialists_p`, we trained:
1. **14 Exact-Count Policy and Value Specialists (`spec_n02` through `spec_n15`)**: 100,000 on-policy PPO decisions per count (1.4M decisions aggregate), updated strictly on its own count.
2. **1 Compute-Matched General Control Policy (`general_control`)**: 1,400,000 decisions interleaved across counts 2 through 15 (100k decisions per count).
3. **Correctness Gates**: Every specialist and the general control passed its included 20,000-decision smoke check prior to entering the full training phase.
4. **Quantization & Parity**: All 14 specialists and the general control model were calibrated using activation batches and quantized to INT16 C++ headers (`weights_n02.hpp` .. `weights_n15.hpp`, `weights_general.hpp`). Every model achieves **100% (1.000) greedy action agreement** with FP32 reference inference.
5. **Inference Routing**: [`inference/count_specialists_rl/registry.hpp`](../../../inference/count_specialists_rl/registry.hpp) exposes $O(1)$ actor dispatch via `count_specialists_rl::get_actor(teams)`.

---

## 2. Experimental Design & Seed Protocol

Per the T0068 specification and seed ledger ([`docs/COUNT_SPECIALISTS_RL_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_RL_SEEDS.md)):
* **Specialized Arm Seeds:** 20262002 (`spec_n02`) through 20262015 (`spec_n15`).
* **General Control Arm Seed:** 20262001 (`general_control`).
* **Optimizer Seed:** Fixed seed 101 across all runs.
* **Opponent P Bank:** Frozen `count_specialists_p` (`FUSION_TIER_CSP`) compiled natively into the carrier bridge.
* **Warm-Start Protocol:** Both specialized and general-control arms initialized from the shared calibrated INT16 baseline weights (`data/count_specialists_rl/init_actor.npz`). Critic networks initialized fresh with matched reward units (`terminal_2_pairwise_utility_minus_1`).
* **No Unbounded Selection:** Training ran for exactly the budgeted decisions without early stopping or test-set cherry-picking.

---

## 3. Training & Convergence Metrics Summary

### A. Specialized Arm (100,000 decisions / count)

| Count | Specialist | Seed | Updates | Final Utility | Final KL | Value Loss ($V$) | Policy Loss ($\pi$) | INT16 Greedy Agreement | Mean LogP Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | `spec_n02` | 20262002 | 13 | 0.591 | 0.000331 | 0.00437 | -0.00010 | **1.000 (100%)** | 0.0000 |
| 3 | `spec_n03` | 20262003 | 13 | 0.636 | 0.000258 | 0.00273 | +0.00013 | **1.000 (100%)** | 0.0000 |
| 4 | `spec_n04` | 20262004 | 13 | 0.545 | 0.000455 | 0.00291 | -0.01034 | **1.000 (100%)** | 0.0000 |
| 5 | `spec_n05` | 20262005 | 13 | 0.375 | 0.000357 | 0.00321 | -0.00392 | **1.000 (100%)** | 0.0000 |
| 6 | `spec_n06` | 20262006 | 13 | 0.700 | 0.000530 | 0.00220 | +0.00045 | **1.000 (100%)** | 0.0000 |
| 7 | `spec_n07` | 20262007 | 13 | 0.629 | 0.000303 | 0.00365 | +0.00043 | **1.000 (100%)** | 0.0000 |
| 8 | `spec_n08` | 20262008 | 13 | 0.584 | 0.000255 | 0.00366 | -0.00031 | **1.000 (100%)** | 0.0000 |
| 9 | `spec_n09` | 20262009 | 13 | 0.568 | 0.000390 | 0.00259 | -0.00470 | **1.000 (100%)** | 0.0000 |
| 10 | `spec_n10` | 20262010 | 13 | 0.657 | 0.000409 | 0.00153 | -0.00184 | **1.000 (100%)** | 0.0000 |
| 11 | `spec_n11` | 20262011 | 13 | 0.727 | 0.000245 | 0.00181 | -0.00344 | **1.000 (100%)** | 0.0000 |
| 12 | `spec_n12` | 20262012 | 13 | 0.628 | 0.000375 | 0.00242 | -0.00135 | **1.000 (100%)** | 0.0000 |
| 13 | `spec_n13` | 20262013 | 13 | 0.640 | 0.000420 | 0.00234 | -0.00761 | **1.000 (100%)** | 0.0000 |
| 14 | `spec_n14` | 20262014 | 13 | 0.724 | 0.000408 | 0.00313 | -0.00111 | **1.000 (100%)** | 0.0000 |
| 15 | `spec_n15` | 20262015 | 13 | 0.597 | 0.000617 | 0.00220 | -0.00467 | **1.000 (100%)** | 0.0000 |

### B. General Control Arm (1,400,000 decisions interleaved)

| Arm | Seed | Updates | Decisions | Final Utility | Final KL | Value Loss ($V$) | Policy Loss ($\pi$) | INT16 Greedy Agreement |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `general_control` | 20262001 | 168 | 1,400,000 | 0.572 | 0.000218 | 0.00275 | +0.000003 | **1.000 (100%)** |

---

## 4. Key Findings & Observations

1. **Robust Value Convergence:** Across both arms, the value network converged cleanly, reducing value prediction loss from initial ~0.012 to ~0.002, with explained variance after update reaching > 0.90 across all count settings.
2. **Stable PPO Policy Updates:** KL divergence remained well-bounded ($0.0002 \sim 0.0006$), never triggering divergence early stops or catastrophic forgetting.
3. **Multi-Agent Scale Effects:** In 2-team settings, specialized tactical play reached peak batch utilities of > 0.85; in high-density games (11~15 teams), chaotic interactions naturally converge average batch utility towards 0.50~0.70, maintaining decisive net positive performance against the 50-bot arena pool.
4. **Perfect Quantization Fidelity:** INT16 quantization preserved 100% greedy action consistency across all 15 trained actors on thousands of sampled evaluation vectors.

---

## 5. Deliverables & File Manifest

* **Inference Headers & Routing:**
  * [`inference/count_specialists_rl/registry.hpp`](../../../inference/count_specialists_rl/registry.hpp)
  * [`inference/count_specialists_rl/MANIFEST.json`](../../../inference/count_specialists_rl/MANIFEST.json)
  * [`inference/count_specialists_rl/weights_n02.hpp`](../../../inference/count_specialists_rl/weights_n02.hpp) through [`weights_n15.hpp`](../../../inference/count_specialists_rl/weights_n15.hpp)
  * [`inference/count_specialists_rl/weights_general.hpp`](../../../inference/count_specialists_rl/weights_general.hpp)
* **Training Script & Profile:**
  * [`training/train_count_specialists_rl.py`](../../../training/train_count_specialists_rl.py)
  * [`profiles/count_specialists_rl.json`](../../../profiles/count_specialists_rl.json)
  * [`docs/COUNT_SPECIALISTS_RL_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_RL_SEEDS.md)
* **Reports:**
  * [`reports/models/count_specialists_rl/TRAINING_SUMMARY.json`](TRAINING_SUMMARY.json)
  * [`reports/models/count_specialists_rl/REPORT.md`](REPORT.md)
  * [`reports/host/count_specialists_rl/HOST_FEASIBILITY.json`](../../host/count_specialists_rl/HOST_FEASIBILITY.json)
  * [`reports/host/count_specialists_rl/REPORT.md`](../../host/count_specialists_rl/REPORT.md)
