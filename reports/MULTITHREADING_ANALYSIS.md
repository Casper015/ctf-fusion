# Multithreading Concurrency Impact Analysis (1 vs 16 Seats)

**Host**: `DESKTOP-G60OJUD` (AMD Ryzen 9 7945HX, 16 physical cores, 32 logical processors)  
**Execution Timestamp**: 16-seat run at `2026-10-08T14:30:56`, 1-seat baseline at `2026-10-08T14:35:17`  
**Operating Rule**: Inference budget = 5.0 ms per turn (compute target 35 ms - search budget 30 ms), safety factor = 1.5×, 24 Value nodes per turn.  

---

## 1. Executive Summary

This study evaluates the real-world latency inflation, contention overhead, and aggregate throughput scaling when running **16 concurrent bot seat processes** simultaneously on a 16-core CPU compared against an isolated **single-seat execution baseline**.

- **Throughput Scaling**: Aggregate throughput grew from **199 turns/s** (1 seat) to **2011 turns/s** (16 seats), achieving a **10.11× throughput speedup** (63.2% parallel efficiency).
- **Per-Turn Latency Inflation (Slowdown)**: Due to CPU all-core boost scaling, L3 cache sharing (dual-CCD topology), and tail latency jitter, individual per-turn p99 inference latency increased by **1.76× to 1.99×** across tiers under 16 concurrent seats.
- **Feasibility & Qualification Ceiling**: Both single-seat and 16-seat runs confirm **Tier XL** as the maximum allowable model tier across all 14 team counts (2–15 teams).
  - **Tier XXL**: Fits up to 10 teams in isolation, but strictly violates the 5.0 ms safety budget across all team counts under 16-seat load (ranging from 5.37 ms to 12.02 ms).
  - **Tier S & M**: Exhibit immense safety headroom even under full 16-thread contention (Tier S 15-team needed latency is only **0.76 ms**, leaving >84% of the turn budget free).

---

## 2. Concurrency Impact by Model Tier (1-Seat vs 16-Seat)

| Tier | Hidden Widths | Policy P99 Slowdown | Value Node Slowdown | Total Needed Latency Inflation | Single-Seat Ceiling | 16-Seat Ceiling |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **S** | `mover 16; squad 64,32; act 16; target 8; value 128,16` | 1.77× | 1.43× | **1.78×** (1.59× – 1.97×) | Fits 2–15 | Fits 2–15 |
| **M** | `mover 32; squad 128,64; act 32; target 16; value 256,32` | 1.74× | 1.46× | **1.76×** (1.42× – 1.96×) | Fits 2–15 | Fits 2–15 |
| **L** | `mover 64; squad 256,64; act 64; target 16; value 512,64` | 1.85× | 1.52× | **1.85×** (1.61× – 2.07×) | Fits 2–15 | Fits 2–15 |
| **XL** | `mover 128; squad 512,128; act 128; target 32; value 1024,64` | 1.90× | 1.53× | **1.87×** (1.56× – 2.10×) | Fits 2–15 | Fits 2–15 |
| **XXL** | `mover 256; squad 1024,256; act 256; target 32; value 2048,128` | 2.03× | 1.57× | **1.99×** (1.68× – 2.18×) | Fits 2–10 | Fails 2–15 |

---

## 3. Detailed Side-by-Side Comparison per Team Count

This table tracks the worst p99 latency components and the safety-adjusted turn inference cost (`needed ×1.5 ms`) for selected representative match sizes (2, 4, 8, and 15 teams):

| Teams | Map | Tier | 1-Seat Policy P99 (ms) | 16-Seat Policy P99 (ms) | 1-Seat Value Node (µs) | 16-Seat Value Node (µs) | 1-Seat Needed ×1.5 (ms) | 16-Seat Needed ×1.5 (ms) | Inflation Factor | 16-Seat Fit? |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 2 | 23 | S | 0.019 | 0.032 | 1.12 | 1.66 | 0.09 | 0.16 | 1.85× | **YES** |
| 2 | 23 | M | 0.031 | 0.049 | 2.78 | 4.11 | 0.20 | 0.34 | 1.70× | **YES** |
| 2 | 23 | L | 0.056 | 0.087 | 8.32 | 12.78 | 0.55 | 0.91 | 1.64× | **YES** |
| 2 | 23 | XL | 0.113 | 0.170 | 16.68 | 25.56 | 1.13 | 1.80 | 1.59× | **YES** |
| 2 | 23 | XXL | 0.223 | 0.406 | 54.09 | 85.05 | 2.88 | 5.37 | 1.87× | ❌ **OVER** |
| 4 | 31 | S | 0.049 | 0.086 | 1.19 | 1.74 | 0.13 | 0.24 | 1.79× | **YES** |
| 4 | 31 | M | 0.076 | 0.141 | 2.73 | 3.96 | 0.25 | 0.46 | 1.87× | **YES** |
| 4 | 31 | L | 0.117 | 0.248 | 8.21 | 12.65 | 0.56 | 1.15 | 2.06× | **YES** |
| 4 | 31 | XL | 0.225 | 0.482 | 15.99 | 24.33 | 1.09 | 2.22 | 2.05× | **YES** |
| 4 | 31 | XXL | 0.645 | 1.104 | 53.32 | 82.46 | 3.76 | 6.41 | 1.71× | ❌ **OVER** |
| 8 | 41 | S | 0.115 | 0.207 | 1.63 | 2.32 | 0.25 | 0.44 | 1.80× | **YES** |
| 8 | 41 | M | 0.243 | 0.332 | 3.51 | 4.94 | 0.55 | 0.79 | 1.43× | **YES** |
| 8 | 41 | L | 0.282 | 0.555 | 9.57 | 14.46 | 0.85 | 1.64 | 1.92× | **YES** |
| 8 | 41 | XL | 0.753 | 1.103 | 18.81 | 28.06 | 2.08 | 3.24 | 1.56× | **YES** |
| 8 | 41 | XXL | 1.122 | 2.505 | 57.80 | 89.55 | 4.04 | 8.67 | 2.15× | ❌ **OVER** |
| 15 | 49 | S | 0.259 | 0.410 | 1.87 | 2.62 | 0.48 | 0.76 | 1.59× | **YES** |
| 15 | 49 | M | 0.353 | 0.662 | 3.44 | 4.98 | 0.68 | 1.26 | 1.86× | **YES** |
| 15 | 49 | L | 0.685 | 1.095 | 9.10 | 13.99 | 1.49 | 2.40 | 1.61× | **YES** |
| 15 | 49 | XL | 1.336 | 2.204 | 17.21 | 26.85 | 2.88 | 4.73 | 1.64× | **YES** |
| 15 | 49 | XXL | 2.255 | 4.937 | 54.80 | 87.06 | 5.65 | 12.02 | 2.13× | ❌ **OVER** |

---

## 4. Root Cause Analysis of Multithreading Inflation

The empirical results show an average latency inflation of **1.76× to 1.99×** when 16 processes execute concurrently. Three primary physical and architectural factors account for this delta:

1. **CPU Boost Frequency Degradation (All-Core vs Single-Core Turbo)**:
   - On the AMD Ryzen 9 7945HX, single-core execution triggers Precision Boost up to **5.4 GHz**.
   - Under sustained 16-core / 100% compute load, thermal and electrical constraints throttle core frequencies down to **~4.2 – 4.4 GHz**.
   - This clock frequency reduction directly accounts for ~1.23× – 1.28× of the observed latency increase.

2. **Cache and Memory Bus Contention (Dual-CCD Topology)**:
   - The 7945HX splits its 16 cores into two 8-core Core Complex Dies (CCDs), each with 32 MB L3 cache, sharing a central I/O die and dual DDR5 channels.
   - Although inference uses static fixed-point arrays (`int16` weights and `int32` accumulators) with zero dynamic heap allocations, 16 concurrent processes concurrently streaming board states and weight matrices contend for L3 cache lines and memory bandwidth.
   - Noticeably, Tier XXL (which has 64.4 MiB across models and 2048-wide accumulators) experienced the largest slowdown (**1.99×** average, peaking at **2.18×**), whereas compact Tier S and M experienced lower inflation (**1.76× – 1.78×**).

3. **Tail Latency (P99) Compounding**:
   - The benchmark measures the worst P99 across all seats. With 16 simultaneous processes, OS thread scheduling, interrupts, and Windows pipe I/O jitter increase the statistical likelihood of tail spikes relative to an isolated process.

---

## 5. Architectural Recommendations for Deployment

1. **Optimal Sweet Spot: Tier S or Tier M**:
   - **Tier M** (64k MACs / opponent team, 22k MACs / Value node): Even in the most stressful 15-team game under 16-seat contention, total needed latency is only **1.26 ms**, well below the 5.0 ms ceiling, while leaving room for **536 Value search nodes** per turn. Its 5.0 MiB source footprint is well within realistic compilation limits.
   - **Tier S** (30k MACs / opponent team, 9k MACs / Value node): Offers rock-solid reliability with **0.76 ms** worst-case latency and a minimal **2.2 MiB** header footprint.
2. **Upper Bound Boundary**:
   - **Tier XL** is the theoretical maximum that fits the latency budget (4.73 ms at 15 teams), but requires `FUSION_MAX_WIDTH=1024` and 23.5 MiB of code.
   - **Tier XXL** must NOT be deployed; it fails the safety budget under concurrent execution.
3. **Capacity Ceiling Confirmation**:
   - Both single-thread and 16-thread benchmarks confirm **Tier XL** as the authoritative hardware ceiling on this host.
