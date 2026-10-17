# Host Feasibility Report: Count Specialists RL (T0068)

**Task ID:** `T0068`  
**Host Machine:** `desktop-g60ojud` (Windows 11, RTX 4080 Laptop GPU, 64GB DDR5)  
**Compiler:** GCC / MinGW-w64 (`-std=c++20 -O2 -static -w`)  
**Carrier:** Deployed carrier variant `agent_v2_rl_csp` (dynamic exact-count PPO actor + refreshed CSP P predictor)  
**Host Limits:** 2,000 ms init, 50 ms per-turn reply limit  
**Preflight Seed:** 20262017 (reserved in [`docs/COUNT_SPECIALISTS_RL_SEEDS.md`](../../../docs/COUNT_SPECIALISTS_RL_SEEDS.md))  

---

## 1. Executive Summary

This report establishes the **exact-eight-worker host feasibility** required by task T0068 prior to launching any formal downstream evaluation match batch.
Host feasibility was verified across two complementary tiers:
1. **Live Game-Level 8-Worker Preflight Gate (`reports/host/count_specialists_rl/PREFLIGHT.json`)**: 64 full official matches (16 maps each across team counts 2, 5, 8, 15) evaluated simultaneously by 8 worker processes running the compiled carrier binary `agent_v2_rl_csp`.
   * **Turn Coverage:** 100% (zero refused turns, zero inconsistent states).
   * **Timing Compliance:** 0 faults, 0 late turns, 0 timeouts.
   * **Max Steady Turn Reply:** 19.80 ms on 15-team games (well inside the 48.0 ms gate and 50.0 ms limit; < 1.6 ms on counts $\le 8$).
   * **Max First Reply:** 57.6 ms on 15-team games (well inside the 1,800 ms gate and 2,000 ms limit).
2. **Native Actor Inference Microbenchmark**: 8 concurrent worker threads evaluating 3,200 turns (48,000 unit forward passes) of the quantized reference16 INT16 actor.
   * **Mean Actor Latency:** 0.0081 ms per turn.
   * **P99 Actor Latency:** 0.0140 ms per turn.

---

## 2. Live 8-Worker Game Preflight Metrics

Across 64 full games (16 games $\times$ 4 counts) against the official reference roster (`profiles/meow_rl_screen.json`):

| Team Count | Games | Turn Coverage | First Reply Max | Steady Reply P50 | Steady Reply P95 | Steady Reply P99 | Steady Reply Max | Limit / Gate | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 16 | 100% | 3.8 ms | 0.236 ms | 0.306 ms | 0.388 ms | 0.432 ms | 50.0 ms / 48.0 ms | **PASSED** |
| 5 | 16 | 100% | 7.4 ms | 0.460 ms | 0.560 ms | 0.627 ms | 0.710 ms | 50.0 ms / 48.0 ms | **PASSED** |
| 8 | 16 | 100% | 22.0 ms | 0.745 ms | 0.906 ms | 1.119 ms | 1.534 ms | 50.0 ms / 48.0 ms | **PASSED** |
| 15 | 16 | 100% | 57.6 ms | 1.796 ms | 2.583 ms | 3.690 ms | 19.802 ms | 50.0 ms / 48.0 ms | **PASSED** |

**Summary Preflight Diagnostics:**
* Total matches evaluated: 64
* Candidate faults / late / dead: **0 / 0 / 0**
* Candidate refused turns: **0**
* Turn coverage: **100%** (all game turns answered with valid orders)
* Total wall time (8 workers): **7.2 seconds**

---

## 3. Isolated Native Actor Microbenchmark Metrics

| Metric | Measured Value | Official Host Limit | Margin / Headroom | Status |
|---|:---:|:---:|:---:|:---:|
| **Init Latency** | `0.01 ms` | `2,000.0 ms` | >200,000x headroom | **PASSED** |
| **Mean Turn Latency** | `0.0081 ms` (8.1 μs) | `50.0 ms` | >6,100x headroom | **PASSED** |
| **P50 Turn Latency** | `0.0070 ms` (7.0 μs) | `50.0 ms` | >7,100x headroom | **PASSED** |
| **P95 Turn Latency** | `0.0130 ms` (13.0 μs) | `50.0 ms` | >3,800x headroom | **PASSED** |
| **P99 Turn Latency** | `0.0140 ms` (14.0 μs) | `50.0 ms` | >3,500x headroom | **PASSED** |
| **Max Turn Latency** | `0.0290 ms` (29.0 μs) | `50.0 ms` | >1,700x headroom | **PASSED** |

---

## 4. Architectural Feasibility Verdict

The integer quantization (`reference16` INT16 with bit-exact SIMD acceleration) and constant memory layout in `inference/count_specialists_rl/` ensure zero dynamic allocations, zero heap contention across worker threads, and negligible CPU overhead.
Under live match concurrency on 8 workers across all team sizes (up to 15 teams), the deployed carrier `agent_v2_rl_csp` operates with substantial safety margins under official 2,000ms / 50ms deadlines.
**Eight-worker host feasibility is firmly established.**
