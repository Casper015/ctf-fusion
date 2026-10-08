# Optimization Attribution and SIMD Benchmark Report

**Host**: `DESKTOP-G60OJUD` (AMD Ryzen 9 7945HX with Radeon Graphics, 32 logical processors, Windows-10-10.0.19045-SP0)  
**Active SIMD Path**: `avx2`  
**Parity Check**: 3,139 checks across scalar, sse2, and avx2 with **0 mismatches**  
**Inference Budget**: 5.0 ms per turn, Safety Factor = 1.5×  

---

## 1. Executive Summary & Attribution Finding

This report evaluates the performance impact of the code updates across two distinct optimization layers:
1. **Data Layout & Memory Restructuring**: Restructuring layer activations to `int16` and transposing the first Value layer matrix (`wT`) so incremental accumulator updates read memory contiguously.
2. **Explicit SIMD Vector Kernels (`inference/simd.hpp`)**: Handcrafted 16-bit multiply-accumulate kernels utilizing AVX2 (`_mm256_madd_epi16`) with fallback paths for SSE2 and NEON.

### Key Findings
- **XXL Viability**: Under full 16-seat concurrent load at 15 teams, total needed inference time for Tier XXL dropped from **12.02 ms down to 2.83 ms**, fully qualifying within the 5.0 ms budget across all 14 team counts.
- **Decomposition of the 4.25× XXL Speedup**:
  - **Data Layout Restructuring**: Contributed **3.29× speedup** (reducing time from 12.02 ms to 3.65 ms). Contiguous memory layout significantly improved auto-vectorization efficiency in the local GCC compiler.
  - **Explicit AVX2 Kernels**: Provided an additional **1.29× speedup** under 16-seat load (3.65 ms → 2.83 ms), and up to **1.98× speedup** under isolated single-seat execution.
- **Rationale for Retaining Explicit SIMD**: Because judge environment compiler versions and optimization flags are unpredictable, compiler auto-vectorization cannot be guaranteed. The explicit AVX2/SSE2 kernels guarantee deterministic SIMD performance regardless of host compiler quirks.

---

## 2. Performance Decomposition at 15 Teams (16 Seats Full Load)

The 15-team match (map size 49×49) represents the maximum competitive computational workload. All times reflect total turn inference needed with the 1.5× safety multiplier.

| Tier | Hidden Widths | Pre-Opt Baseline | New Layout (Scalar) | New Layout + AVX2 | Layout Speedup | AVX2 Speedup | Combined Speedup | 5ms Budget Status |
|:---:|---|---:|---:|---:|---:|---:|---:|:---:|
| **S** | `squad=64,32; value=128,16` | 0.76 ms | 0.53 ms | **0.50 ms** | 1.45× | 1.06× | **1.53×** | **PASSED** |
| **M** | `squad=128,64; value=256,32` | 1.26 ms | 0.70 ms | **0.64 ms** | 1.80× | 1.09× | **1.96×** | **PASSED** |
| **L** | `squad=256,64; value=512,64` | 2.40 ms | 0.99 ms | **0.85 ms** | 2.41× | 1.17× | **2.83×** | **PASSED** |
| **XL** | `squad=512,128; value=1024,64` | 4.73 ms | 1.70 ms | **1.36 ms** | 2.78× | 1.25× | **3.48×** | **PASSED** |
| **XXL** | `squad=1024,256; value=2048,128` | 12.02 ms | 3.65 ms | **2.83 ms** | 3.29× | 1.29× | **4.25×** | **PASSED** |

---

## 3. Isolated Single-Seat Execution: AVX2 vs Scalar (15 Teams)

Under single-seat execution without process contention, the raw vectorization throughput of the explicit AVX2 kernel becomes prominent on wider hidden dimensions:

| Tier | Policy p99 (Scalar) | Policy p99 (AVX2) | Policy Speedup | Value Node (Scalar) | Value Node (AVX2) | Value Speedup | Total Needed (Scalar) | Total Needed (AVX2) | Total Speedup |
|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **S** | 0.181 ms | 0.165 ms | **1.10×** | 1.29 µs | 1.15 µs | **1.12×** | 0.34 ms | 0.30 ms | **1.11×** |
| **M** | 0.219 ms | 0.199 ms | **1.10×** | 1.77 µs | 1.40 µs | **1.27×** | 0.42 ms | 0.36 ms | **1.14×** |
| **L** | 0.263 ms | 0.265 ms | **0.99×** | 3.05 µs | 2.08 µs | **1.46×** | 0.52 ms | 0.51 ms | **1.02×** |
| **XL** | 0.585 ms | 0.355 ms | **1.65×** | 5.16 µs | 3.17 µs | **1.63×** | 1.18 ms | 0.69 ms | **1.72×** |
| **XXL** | 1.154 ms | 0.604 ms | **1.91×** | 13.91 µs | 8.77 µs | **1.59×** | 2.55 ms | 1.29 ms | **1.98×** |

---

## 4. Cross-Team Scaling & Measurement Reliability

> [!NOTE]
> **Measurement Noise on Small Team Counts**: At 2–4 teams, total inference time is on the order of 0.03–0.10 ms. At these microsecond scales, OS scheduling, timer granularity, and process launch jitter dominate the measurement, occasionally producing noisy inversion ratios (e.g. 0.7×–0.9×). Authoritative capacity decisions should rely on larger match configurations (6–15 teams).

| Teams | Map | Tier | Pre-Opt Baseline ms | New Layout + AVX2 ms | Net Speedup | 16-Seat Fits? | Value Nodes Affordable |
|---:|:---:|:---:|---:|---:|---:|:---:|---:|
| 6 | 41 | M | 0.62 ms | **0.29 ms** | **2.16×** | YES | 2,281 |
| 6 | 41 | XL | 2.62 ms | **0.69 ms** | **3.82×** | YES | 697 |
| 6 | 41 | XXL | 7.36 ms | **1.57 ms** | **4.68×** | YES | 207 |
| 8 | 41 | M | 0.79 ms | **0.44 ms** | **1.81×** | YES | 1,689 |
| 8 | 41 | XL | 3.24 ms | **0.82 ms** | **3.93×** | YES | 615 |
| 8 | 41 | XXL | 8.67 ms | **1.85 ms** | **4.69×** | YES | 192 |
| 10 | 41 | M | 0.93 ms | **0.70 ms** | **1.32×** | YES | 1,546 |
| 10 | 41 | XL | 3.63 ms | **1.03 ms** | **3.52×** | YES | 561 |
| 10 | 41 | XXL | 9.50 ms | **2.11 ms** | **4.50×** | YES | 190 |
| 12 | 49 | M | 1.03 ms | **0.51 ms** | **2.03×** | YES | 1,699 |
| 12 | 49 | XL | 3.92 ms | **1.14 ms** | **3.44×** | YES | 569 |
| 12 | 49 | XXL | 10.33 ms | **2.36 ms** | **4.38×** | YES | 176 |
| 15 | 49 | M | 1.26 ms | **0.64 ms** | **1.96×** | YES | 1,433 |
| 15 | 49 | XL | 4.73 ms | **1.36 ms** | **3.48×** | YES | 523 |
| 15 | 49 | XXL | 12.02 ms | **2.83 ms** | **4.25×** | YES | 137 |

---

## 5. Technical Rigor & Mathematical Properties

1. **Mathematical Basis of Bit-Exact Parity**:
   - AVX2 processes 16 `int16` products concurrently using horizontal adds (`_mm256_madd_epi16`), which computes sums in a different pairwise order than the naive sequential loop.
   - Bit-exact identity is guaranteed because integer addition is strictly commutative and associative **under the precondition of no arithmetic overflow**.
   - The weight quantization exporter formally proves an accumulator bound of `[-2^30, 2^30]`, precluding any 32-bit signed overflow. Thus, both paths yield identical results down to the last bit.
2. **Scope of Benchmark Measurements**:
   - These benchmarks profile isolated neural network inference (features → forward pass → accumulator update).
   - They do not measure full turn bot overhead (stream parsing, search rollouts, heuristic move ordering, formatting).
   - Cache hit/miss rates and CPU core frequencies are inferences, not direct PMU hardware counter telemetry.
3. **Deployment Model Selection**:
   - Because Tier XXL now operates safely within the 5.0 ms ceiling across all team counts (2.83 ms at 15 teams), deployment should not be arbitrarily restricted by source code size.
   - The final model selection will be decided by empirical offline log-loss and Bradley-Terry playing strength gain per millisecond once trained models are available.
