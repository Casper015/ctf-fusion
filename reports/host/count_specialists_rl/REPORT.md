# Host Feasibility Report: Count Specialists RL (T0068)

**Task ID:** `T0068`  
**Host Machine:** `desktop-g60ojud` (Windows 11, RTX 4080 Laptop GPU, 64GB DDR5)  
**Compiler:** GCC / MinGW-w64 (`-std=c++20 -O3 -w`)  
**Architecture:** INT16 MLP Actor (`fusion::Net`, 112 -> 64 -> 32 -> 1, reference16 INT16)  
**Routing Registry:** [`inference/count_specialists_rl/registry.hpp`](../../../inference/count_specialists_rl/registry.hpp)  
**Host Limits:** 2,000 ms init, 50 ms per-turn reply limit  

---

## 1. Executive Summary

This report establishes the **exact-eight-worker host feasibility** required by task T0068 prior to launching any formal downstream evaluation match batch.
Under 8 concurrent worker threads evaluating complete 400-turn game streams on 15-team games (the maximum tournament count):

* **Initialization Latency:** `< 0.01 ms` (well within the 2,000 ms initialization budget).
* **Per-Turn Inference Latency:** **0.0081 ms mean**, **0.0140 ms p99**, and **0.0290 ms max** across 3,200 evaluated turns.
* **Turn Budget Headroom:** Over **3,500x faster** than the official 50 ms turn deadline.
* **Routing Efficiency:** True $O(1)$ dispatch by exact team count in `count_specialists_rl::get_actor(teams)`, evaluating only the assigned policy without invoking remaining specialists.

---

## 2. Benchmark Configuration & Methodology

The microbenchmark (`build/bench_rl_host.exe`) simulates realistic turn workloads under full host saturation:
* **Concurrency:** 8 parallel worker threads executing simultaneously without locks.
* **Evaluated Model:** 15-team specialist (`fusion::models::rl_n15::kActor`) and $O(1)$ registry routing.
* **Per-Turn Compute:** 3 units $\times$ 5 move choices = 15 complete INT16 forward passes through the 112-input actor, followed by 125-joint-action score aggregation.
* **Turns Per Worker:** 400 turns (1 full match equivalent per thread).
* **Total Evaluated Turns:** 3,200 turns (48,000 unit-move forward passes).

---

## 3. Measured Host Latency Metrics

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
**Eight-worker host feasibility is firmly established.**
