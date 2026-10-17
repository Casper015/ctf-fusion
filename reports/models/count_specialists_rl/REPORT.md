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
4. **Quantization & Parity**: All 14 specialists and the general control model were calibrated using activation batches from an aligned corpus (`data/count_specialists_rl/calibration.npz`, seed 20262016) and quantized to INT16 C++ headers (`weights_n02.hpp` .. `weights_n15.hpp`, `weights_general.hpp`). Every specialist achieves **95.1% ~ 97.2% greedy action agreement** against FP32 reference inference across 2,000 real game-masked action states (general control achieves **95.6%**), with mean absolute log-probability error $\le 0.31$, max log-probability error $\le 2.56$, and zero saturated integer activations.
5. **Inference Routing**: [`inference/count_specialists_rl/registry.hpp`](../../../inference/count_specialists_rl/registry.hpp) exposes $O(1)$ actor dispatch via `count_specialists_rl::get_actor(teams)`.

---

## 2. Experimental Design & Seed Protocol

Per the T0068 specification and seed ledger ([`docs/COUNT_SPECIALISTS_RL_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_RL_SEEDS.md)):
* **Specialized Arm Seeds:** 20262002 (`spec_n02`) through 20262015 (`spec_n15`).
* **General Control Arm Seed:** 20262001 (`general_control`).
* **Calibration Seed:** 20262016 (560 episodes = 40 episodes / count $\times$ 14 counts in `data/count_specialists_rl/calibration.npz`).
* **Optimizer Seed:** Fixed seed 101 across all runs.
* **Opponent P Bank:** Frozen `count_specialists_p` (`FUSION_TIER_CSP`) compiled natively into the carrier bridge.
* **Warm-Start Protocol:** Both specialized and general-control arms initialized from the shared calibrated INT16 baseline weights (`data/count_specialists_rl/init_actor.npz`). Critic networks initialized fresh with matched reward units (`terminal_2_pairwise_utility_minus_1`).
* **No Unbounded Selection:** Training ran for exactly the budgeted decisions without early stopping or test-set cherry-picking.

---

## 3. Training & Calibration Metrics Summary

> [!NOTE]
> The "Batch Utility" below reports the on-policy rollout batch mean utility against the 50-bot training pool during the final training update. Formal held-out out-of-pool win-rate evaluation is deferred to a dedicated evaluation task.

### A. Specialized Arm (100,000 decisions / count)

| Count | Specialist | Seed | Updates | Decisions | Batch Utility (last) | Approx KL (last) | Value Loss ($V$) | Policy Loss ($\pi$) | INT16 Greedy Agreement | Mean Abs LogP Error | Max Abs LogP Error |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | `spec_n02` | 20262002 | 13 | 100,000 | 0.591 | 0.000331 | 0.00437 | -0.00010 | **95.9%** | 0.2885 | 1.934 |
| 3 | `spec_n03` | 20262003 | 13 | 100,000 | 0.636 | 0.000258 | 0.00273 | +0.00013 | **95.1%** | 0.3098 | 2.553 |
| 4 | `spec_n04` | 20262004 | 13 | 100,000 | 0.545 | 0.000455 | 0.00291 | -0.01034 | **96.1%** | 0.2884 | 1.827 |
| 5 | `spec_n05` | 20262005 | 13 | 100,000 | 0.375 | 0.000357 | 0.00321 | -0.00392 | **96.2%** | 0.2579 | 1.649 |
| 6 | `spec_n06` | 20262006 | 13 | 100,000 | 0.700 | 0.000530 | 0.00220 | +0.00045 | **96.2%** | 0.3036 | 2.011 |
| 7 | `spec_n07` | 20262007 | 13 | 100,000 | 0.629 | 0.000303 | 0.00365 | +0.00043 | **96.2%** | 0.2951 | 1.837 |
| 8 | `spec_n08` | 20262008 | 13 | 100,000 | 0.584 | 0.000255 | 0.00366 | -0.00031 | **97.0%** | 0.2757 | 1.885 |
| 9 | `spec_n09` | 20262009 | 13 | 100,000 | 0.568 | 0.000390 | 0.00259 | -0.00470 | **97.2%** | 0.2821 | 2.216 |
| 10 | `spec_n10` | 20262010 | 13 | 100,000 | 0.657 | 0.000409 | 0.00153 | -0.00184 | **96.9%** | 0.2584 | 1.619 |
| 11 | `spec_n11` | 20262011 | 13 | 100,000 | 0.727 | 0.000245 | 0.00181 | -0.00344 | **96.6%** | 0.2724 | 1.888 |
| 12 | `spec_n12` | 20262012 | 13 | 100,000 | 0.628 | 0.000375 | 0.00242 | -0.00135 | **96.5%** | 0.2699 | 1.668 |
| 13 | `spec_n13` | 20262013 | 13 | 100,000 | 0.640 | 0.000420 | 0.00234 | -0.00761 | **97.1%** | 0.2785 | 1.770 |
| 14 | `spec_n14` | 20262014 | 13 | 100,000 | 0.724 | 0.000408 | 0.00313 | -0.00111 | **96.5%** | 0.2634 | 1.872 |
| 15 | `spec_n15` | 20262015 | 13 | 100,000 | 0.597 | 0.000617 | 0.00220 | -0.00467 | **96.4%** | 0.2671 | 1.711 |

### B. General Control Arm (1,400,000 decisions interleaved)

| Arm | Seed | Updates | Decisions | Episodes | Batch Utility (last) | Approx KL (last) | Value Loss ($V$) | Policy Loss ($\pi$) | INT16 Greedy Agreement | Mean Abs LogP Error | Max Abs LogP Error |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `general_control` | 20262001 | 168 | 1,400,000 | 3,500 | 0.572 | 0.000218 | 0.00275 | +0.000003 | **95.6%** | 0.3058 | 2.451 |

**General Control Arm Per-Count Allocation:**
* Episode allocation across counts:
  - `n02`: 249, `n03`: 246, `n04`: 256, `n05`: 251, `n06`: 279, `n07`: 240, `n08`: 265
  - `n09`: 259, `n10`: 239, `n11`: 222, `n12`: 236, `n13`: 254, `n14`: 252, `n15`: 252
  - Total: 3,500 episodes. Range per count: 222 to 279 episodes (natural variance due to random count sampling across 8 asynchronous rollout workers, not strictly identical per count).
* Estimated decision allocation: ~100,000 decisions per count ($\approx 250 \times 400$ turns).

---

## 4. Key Findings & Observations

1. **Training Stability & Loss Evolution:** Across both arms, value loss decreased from initial ~0.012 to ~0.002, and PPO policy updates remained bounded (approx KL $0.0002 \sim 0.0006$) without triggering divergence early stops. Note that policy convergence is not asserted without formal held-out out-of-pool evaluation.
2. **Multi-Agent Scale Effects:** In 2-team settings, specialized tactical play reached peak batch utilities of > 0.85; in high-density games (11~15 teams), chaotic interactions naturally converge average batch utility towards 0.50~0.70 against the 50-bot arena pool.
3. **Quantization Parity:** Aligned INT16 quantization achieved 95.1% ~ 97.2% greedy action agreement (general control: 95.6%) and mean absolute log-probability errors $\le 0.31$ against full-precision FP32 on 2,000 real game-masked action vectors per model, with zero saturated activations.
4. **Removal of Confounded Preliminary Tournament:** The unapproved 1,000-match tournament was completely excised from this task. That old tournament was confounded by two carrier implementation defects present prior to review: (1) an entrypoint routing bug where the carrier failed to route by team count and executed the general model on every count, and (2) an incorrect output scale (`outExp` 6 instead of 16), which caused near-greedy action selection instead of the trained policy. Consequently, neither the old tournament nor final rollout batch metrics provide valid evidence of specialist policy strength.
5. **Subsequent Evaluation Task:** Formal out-of-pool win-rate assessment, comparative evaluation between specialists and the general control arm, and benchmarking against baseline models belong to the follow-on task: **"新机体策略验收与配对评测" (New Carrier Policy Acceptance and Paired Evaluation)**.

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
