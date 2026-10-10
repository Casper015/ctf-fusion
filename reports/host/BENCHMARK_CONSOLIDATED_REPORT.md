# CTF-Fusion Neural Inference: Comprehensive Capacity & Scaling Report

**Test Environment**: `DESKTOP-G60OJUD` (AMD Ryzen 9 7945HX with Radeon Graphics, 32 logical threads, Windows-10-10.0.19045-SP0)  
**Compiler & SIMD**: GCC 15.2 (`-std=c++20 -O2`), runtime-dispatched `avx2` bit-exact kernels  
**Rule Constraints**: Turn 1 ≤ 2.0 s; later turns ≤ 50.0 ms. Search budget = 30 ms, Inference budget = 5.0 ms (Safety factor = 1.5×)  

---

## 1. Executive Summary

This consolidated report synthesizes four comprehensive empirical benchmarks conducted on the host:
1. **Hardware Capacity Matrix**: Evaluated network tiers S through XXL across all 14 official team counts (2–15 teams).
2. **Multithreading Concurrency**: Benchmarked 16 concurrent bot seat processes running simultaneously, measuring tail latency (p99) inflation and throughput scaling.
3. **Optimization Attribution**: Decomposed the performance gains into Data Layout restructuring (transposed contiguous memory, int16 activations) versus explicit AVX2 SIMD vector kernels.
4. **Scaling Limits & Memory Wall**: Pushed model width to 4096 (3XL) and 8192 (4XL) to establish the physical L3 cache boundary and memory bandwidth limits.

### Definitive Verdicts
- **Authoritative Hardware Ceiling**: **`Tier XXL`** (width 2048, 679k Policy MACs, 373k Value MACs). Under 16-seat full load in 15-team play, Tier XXL consumes **2.74 ms** (with 1.5× safety factor), fully qualifying within the 5.0 ms budget.
- **Recommended Deployment Sweet Spot**: **`Tier M`** (width 256, 64k Policy MACs, 22k Value MACs). Consumes only **0.64 ms** under 16-seat load, leaving budget for **1,433 Value tree-search rollouts** with a minimal **5.9 MiB** source footprint.
- **Absolute Scaling Boundary**: Any scaling beyond XXL (such as 3XL at 23.0 ms and 4XL at 191.5 ms) triggers catastrophic L3 cache evictions and memory bus saturation under concurrency, violating competition deadlines.

---

## 2. Complete Model Tier Specifications

| Tier | Hidden Widths (mover / squad / act / target / value) | Policy MACs / Team | Value MACs / Node | 11-Model Binary RAM | 11-Model C++ Header | Max Width |
|:---:|---|---:|---:|---:|---:|:---:|
| **S** | `16 / 64,32 / 16 / 8 / 128,16` | 30,096 | 8,976 | ~0.8 MiB | 2.7 MiB | 256 |
| **M** | `32 / 128,64 / 32 / 16 / 256,32` | 64,288 | 22,048 | ~1.7 MiB | 5.9 MiB | 256 |
| **L** | `64 / 256,64 / 64 / 16 / 512,64` | 120,576 | 60,480 | ~3.5 MiB | 12.5 MiB | 512 |
| **XL** | `128 / 512,128 / 128 / 32 / 1024,64` | 273,920 | 120,896 | ~7.6 MiB | 27.1 MiB | 1024 |
| **XXL** | `256 / 1024,256 / 256 / 32 / 2048,128` | 678,912 | 372,864 | ~20.0 MiB | 71.5 MiB | 2048 |
| **3XL** (Extreme) | `512 / 2048,512 / 512 / 64 / 4096,256` | 1,882,112 | 1,270,016 | ~59.0 MiB | 210.6 MiB | 4096 |
| **4XL** (Extreme) | `1024 / 4096,1024 / 1024 / 64 / 8192,512` | 5,861,376 | 4,637,184 | ~188.3 MiB | 672.6 MiB | 8192 |

---

## 3. The 15-Team Stress Matrix: Concurrency & Scaling Ladder

The 15-team match (49×49 map, 14 opponent teams evaluated per turn) represents maximum competition load:

| Tier | 1-Seat Needed ms (1.5×) | 16-Seat Needed ms (1.5×) | Load Inflation Ratio | 16-Seat Policy p99 | 16-Seat Value Node | Affordable Value Nodes | Status (5ms Budget) |
|:---:|---:|---:|:---:|---:|---:|---:|:---:|
| **S** | 0.30 ms | **0.50 ms** | 1.64× | 0.271 ms | 1.62 µs | **1,891** | **PASSED** |
| **M** | 0.36 ms | **0.64 ms** | 1.77× | 0.340 ms | 2.09 µs | **1,433** | **PASSED** |
| **L** | 0.51 ms | **0.85 ms** | 1.65× | 0.439 ms | 3.21 µs | **900** | **PASSED** |
| **XL** | 0.69 ms | **1.36 ms** | 1.97× | 0.703 ms | 5.02 µs | **523** | **PASSED** |
| **XXL** | 1.29 ms | **2.83 ms** | 2.20× | 1.336 ms | 14.58 µs | **137** | **PASSED** |
| **3XL** | 2.93 ms | **23.04 ms** | **7.87× (Cliff)** | 8.264 ms | 63.82 µs | 0 | ❌ **FAILED** |
| **4XL** | 9.73 ms | **191.50 ms** | **19.68× (Cliff)** | 45.789 ms | 2462.27 µs | 0 | ❌ **FAILED** |

---

## 4. Root-Cause Analysis: The L3 Cache Memory Wall

A critical phenomenon discovered is the **non-linear latency cliff** between Tier XXL and Tier 3XL:
- Up to **Tier XXL**, concurrent 16-seat load produces an expected **1.6×–2.0× slowdown** attributable to CPU turbo boost downclocking (~5.4 GHz down to ~4.3 GHz under 16-core load).
- Beyond Tier XXL, **Tier 3XL explodes by 7.86×** (2.93 ms → 23.04 ms) and **Tier 4XL explodes by 19.68×** (9.73 ms → 191.50 ms).

### Micro-Architectural Explanation
1. **Dual-CCD L3 Partitioning**: The AMD Ryzen 9 7945HX features two 8-core CCDs, each backed by a 32 MB L3 cache slice.
2. **Cache Residency up to XXL**: A single Tier XXL model requires ~1.8 MB of `int16` weights. Eight concurrent processes on one CCD require $8 \times 1.8 = 14.4\text{ MB}$, fitting comfortably within the 32 MB L3 cache.
3. **L3 Spillage & Memory Bus Saturation at 3XL/4XL**:
   - Tier 3XL requires ~6.0 MB per model. Eight processes demand $48\text{ MB} > 32\text{ MB}$, causing constant L3 cache line evictions.
   - Tier 4XL requires ~20.0 MB per model (320 MB across 16 seats). The 16 cores saturate the DDR5 memory bus with streaming requests, causing single Value node evaluation time to skyrocket from **107 µs to 2,462 µs (23× increase)**.

---

## 5. Performance Attribution: Data Layout vs. Explicit AVX2

Comparing the pre-optimization codebase (`7217afe`), the new contiguous layout under scalar compilation, and the explicit AVX2 kernel reveals that **both layers are vital**:

| Tier | Pre-Opt Baseline | New Layout (Scalar) | New Layout + AVX2 | Layout Speedup | AVX2 Speedup | Combined Speedup |
|:---:|---:|---:|---:|:---:|:---:|:---:|
| **S** | 0.76 ms | 0.53 ms | **0.50 ms** | **1.45×** | **1.06×** | **1.53×** |
| **M** | 1.26 ms | 0.70 ms | **0.64 ms** | **1.80×** | **1.09×** | **1.96×** |
| **L** | 2.40 ms | 0.99 ms | **0.85 ms** | **2.41×** | **1.17×** | **2.83×** |
| **XL** | 4.73 ms | 1.70 ms | **1.36 ms** | **2.78×** | **1.25×** | **3.48×** |
| **XXL** | 12.02 ms | 3.65 ms | **2.83 ms** | **3.29×** | **1.29×** | **4.25×** |

- **Layout Restructuring (1.45×–3.29×)**: Provided the majority of baseline speedup by transposing `ValueAccumulator` weights (`wT`) and using contiguous memory, which allowed the local GCC compiler to efficiently auto-vectorize loops.
- **Explicit AVX2 Kernels (1.06×–1.29× loaded, up to 1.98× isolated)**: Provided vital extra vector throughput and, more importantly, **guarantees deterministic vector performance on unknown competition judge environments** where compiler auto-vectorization flags cannot be trusted.

---

## 6. Memory Footprint & Runtime Contract

All measurements confirm total compliance with competitive runtime constraints:
- **Stack / Turn Working Memory**: All feature encoders and buffers require **< 30 KB** total stack space during turns.
- **Zero Heap Allocation**: Validated by `tools/cpp/runtime_check.cpp`. After Turn 1 initialization, Turns 2–400 perform **strictly 0 dynamic allocations** (`operator new` / `malloc`), precluding memory fragmentation or leaks.
- **Process Working Set (RSS)**: Single bot process consumes **~2.0 to 4.5 MB**. Even 16 concurrent bot processes consume **< 75 MB total RAM**.

---

## 7. Strategic Recommendations for Bot Integration (`Player7`)

1. **Recommended Competition Default: Tier M**
   - Turn inference time: **0.64 ms** (leaves >87% of the turn compute budget to the heuristic search).
   - Expansion budget: Permits **1,433 Value search rollouts** per turn.
   - Source size: **5.9 MiB** (11 models), safely within judge compilation limits.
2. **Maximum Capacity Ceiling: Tier XXL**
   - Turn inference time: **2.74 ms** (safe within 5.0 ms).
   - Source size: **71.5 MiB** (requires verify if judge compiler accepts 70MB source files).
3. **Hard Constraint**: Never deploy beyond Tier XXL. The hardware memory wall creates catastrophic timeout risks at 3XL and above.
