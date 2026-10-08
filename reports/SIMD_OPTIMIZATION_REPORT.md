# SIMD Kernel and Memory Architecture Optimization Report

**Target Host**: `DESKTOP-G60OJUD` (AMD Ryzen 9 7945HX, 16 Cores / 32 Threads, Windows)  
**SIMD Path Active**: `avx2` (AVX2 256-bit bit-exact integer vector kernels)  
**Concurrency Level**: 16 concurrent bot seat processes (`--seats 16`)  
**Turn Inference Budget**: 5.0 ms (Target 35 ms - Search 30 ms), Safety Factor = 1.5×, 24 Value nodes/turn  

---

## 1. Executive Summary

With the integration of the SIMD multiply-add optimization package (`inference/simd.hpp`) and contiguous memory transposition for `ValueAccumulator` (`inference/value.hpp`), the inference performance across all model tiers experienced dramatic speedups:

- **Hardware Ceiling Breakthrough (XL → XXL)**: In the original scalar architecture, **Tier XXL** severely violated the 5.0 ms deadline across all team counts under 16-thread load (e.g. 12.02 ms at 15 teams). With AVX2 SIMD kernels and transposed value accumulators, **Tier XXL now easily satisfies the budget across all 14 team counts (2.83 ms at 15 teams)**, achieving a **4.25× speedup**.
- **Speedup Proportional to Network Width**: Wider network layers utilize SIMD registers and vector lanes with maximal efficiency:
  - **Tier S** (width 64): **1.53× speedup** (0.76 ms → 0.50 ms)
  - **Tier M** (width 128): **1.96× speedup** (1.26 ms → 0.64 ms)
  - **Tier L** (width 256): **2.83× speedup** (2.40 ms → 0.85 ms)
  - **Tier XL** (width 512): **3.48× speedup** (4.73 ms → 1.36 ms)
  - **Tier XXL** (width 1024/2048): **4.25× speedup** (12.02 ms → 2.83 ms)
- **Bit-Exact Parity**: The integer dot product operations maintain identical int32 accumulation order, rigorously verified by `tools/kernel_check.cpp` (3,139 tests passed with 0 mismatches). No floating-point rounding divergence exists.

---

## 2. Architectural Optimizations Introduced

1. **AVX2 / SSE2 16-bit Multiply-Add Vector Kernels (`inference/simd.hpp`)**:
   - Utilizes `_mm256_madd_epi16` to perform 16 parallel signed 16-bit multiplications and pair-wise additions into 32-bit accumulators in a single cycle.
   - Dispatches at runtime via GCC/Clang target attributes (`__attribute__((target("avx2")))`) with zero compilation flag overhead and transparent SSE2/NEON/Scalar fallbacks.
2. **Transposed Weight Matrix for Value Accumulator (`inference/value.hpp`)**:
   - The incremental board value accumulator (`ValueAccumulator::update`) modifies dot products whenever feature deltas occur.
   - By transposing the weight matrix (`wT`), accumulator updates access memory along contiguous cache lines (`wT + i * out`) instead of strided column leaps (`w + o * in + i`). This eliminated L1/L2 cache misses during tree search node rollouts.
3. **Single Buffer Re-use & Zero Allocation**:
   - Fixed stack buffers (`kMaxWidth`) avoid heap allocations on hot inference paths.

---

## 3. Comprehensive Benchmark Comparison (Original Scalar vs Optimized SIMD)

### 15 Teams (Maximum Competition Complexity, Map 49×49, 16 Concurrent Seats)

| Tier | Width Profile | Original Needed ms (1.5×) | Optimized Needed ms (1.5×) | Speedup Ratio | Original Value µs | Optimized Value µs | Value Speedup | Affordable Value Nodes | Qualification |
|:---:|---|---:|---:|:---:|---:|---:|:---:|---:|:---:|
| **S** | `squad=64,32; value=128,16` | 0.76 ms | **0.50 ms** | **1.53×** | 2.62 µs | **1.62 µs** | **1.62×** | **1,891** | **PASSED** |
| **M** | `squad=128,64; value=256,32` | 1.26 ms | **0.64 ms** | **1.96×** | 4.98 µs | **2.09 µs** | **2.38×** | **1,433** | **PASSED** |
| **L** | `squad=256,64; value=512,64` | 2.40 ms | **0.85 ms** | **2.83×** | 13.99 µs | **3.21 µs** | **4.35×** | **900** | **PASSED** |
| **XL** | `squad=512,128; value=1024,64` | 4.73 ms | **1.36 ms** | **3.48×** | 26.85 µs | **5.02 µs** | **5.34×** | **523** | **PASSED** |
| **XXL** | `squad=1024,256; value=2048,128` | 12.02 ms | **2.83 ms** | **4.25×** | 87.06 µs | **14.58 µs** | **5.97×** | **137** | **PASSED (XXL)** |

### Cross-Team Comparison (Tiers S, M, XL, XXL across 2, 4, 8, 12, 15 Teams)

| Teams | Map | Tier | Original Needed ms | Optimized Needed ms | Total Speedup | Original Policy p99 | Optimized Policy p99 | Policy Speedup | Status |
|---:|:---:|:---:|---:|---:|:---:|---:|---:|:---:|:---:|
| 2 | 23 | S | 0.16 ms | **0.08 ms** | **2.06×** | 0.032 ms | **0.025 ms** | **1.28×** | PASSED |
| 2 | 23 | M | 0.34 ms | **0.12 ms** | **2.88×** | 0.049 ms | **0.028 ms** | **1.75×** | PASSED |
| 2 | 23 | XL | 1.80 ms | **0.35 ms** | **5.09×** | 0.170 ms | **0.059 ms** | **2.88×** | PASSED |
| 2 | 23 | XXL | 5.37 ms | **1.02 ms** | **5.25×** | 0.406 ms | **0.149 ms** | **2.72×** | PASSED |
| 4 | 31 | S | 0.24 ms | **0.14 ms** | **1.66×** | 0.086 ms | **0.059 ms** | **1.46×** | PASSED |
| 4 | 31 | M | 0.46 ms | **0.18 ms** | **2.51×** | 0.141 ms | **0.074 ms** | **1.91×** | PASSED |
| 4 | 31 | XL | 2.22 ms | **0.48 ms** | **4.65×** | 0.482 ms | **0.160 ms** | **3.01×** | PASSED |
| 4 | 31 | XXL | 6.41 ms | **1.19 ms** | **5.38×** | 1.104 ms | **0.318 ms** | **3.47×** | PASSED |
| 8 | 41 | S | 0.44 ms | **0.33 ms** | **1.32×** | 0.207 ms | **0.160 ms** | **1.29×** | PASSED |
| 8 | 41 | M | 0.79 ms | **0.44 ms** | **1.81×** | 0.332 ms | **0.209 ms** | **1.59×** | PASSED |
| 8 | 41 | XL | 3.24 ms | **0.82 ms** | **3.93×** | 1.103 ms | **0.365 ms** | **3.02×** | PASSED |
| 8 | 41 | XXL | 8.67 ms | **1.85 ms** | **4.69×** | 2.505 ms | **0.707 ms** | **3.54×** | PASSED |
| 12 | 49 | S | 0.62 ms | **0.41 ms** | **1.52×** | 0.321 ms | **0.212 ms** | **1.51×** | PASSED |
| 12 | 49 | M | 1.03 ms | **0.51 ms** | **2.03×** | 0.524 ms | **0.268 ms** | **1.96×** | PASSED |
| 12 | 49 | XL | 3.92 ms | **1.14 ms** | **3.44×** | 1.718 ms | **0.576 ms** | **2.98×** | PASSED |
| 12 | 49 | XXL | 10.33 ms | **2.36 ms** | **4.38×** | 3.902 ms | **1.058 ms** | **3.69×** | PASSED |
| 15 | 49 | S | 0.76 ms | **0.50 ms** | **1.53×** | 0.410 ms | **0.271 ms** | **1.51×** | PASSED |
| 15 | 49 | M | 1.26 ms | **0.64 ms** | **1.96×** | 0.662 ms | **0.340 ms** | **1.95×** | PASSED |
| 15 | 49 | XL | 4.73 ms | **1.36 ms** | **3.48×** | 2.204 ms | **0.703 ms** | **3.14×** | PASSED |
| 15 | 49 | XXL | 12.02 ms | **2.83 ms** | **4.25×** | 4.937 ms | **1.336 ms** | **3.70×** | PASSED |

---

## 4. Value Search Expansion Headroom

Because value evaluation was accelerated from **87.06 µs down to 14.58 µs** per node on Tier XXL, and from **4.98 µs down to 2.09 µs** on Tier M:

- **Tier M Expansion**: Can now evaluate **1,433 search nodes** per turn under full 16-thread load (up from 536 nodes).
- **Tier S Expansion**: Can evaluate **1,891 search nodes** per turn (up from 1,116 nodes).
- **Tier XXL Viability**: Unlocks deep 1-2 ply search rollouts even with the largest 2048-wide value network.

---

## 5. Engineering Conclusion & Recommendation

1. **Recommended Default: Tier M**:
   - At 15 teams, 16 concurrent seats take only **0.64 ms** total inference time.
   - C++ header footprint is **5.9 MiB**, compiling rapidly without triggering judge file-size alarms.
   - Provides **1,433 value search rollouts** of budget headroom.
2. **High-Capacity Option: Tier XL / XXL**:
   - The SIMD update completely eliminates compute bottlenecks for Tier XL (1.36 ms) and Tier XXL (2.83 ms).
   - If the judge environment permits a 27–71 MiB binary source inclusion, Tier XL or XXL can be deployed directly with zero timeout risk.