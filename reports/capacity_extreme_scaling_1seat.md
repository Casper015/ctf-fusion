# Capacity benchmark: DESKTOP-G60OJUD

2026-10-08T16:13:53

CPU AMD Ryzen 9 7945HX with Radeon Graphics, 32 logical processors, Windows-10-10.0.19045-SP0. SIMD kernels: avx2.

Budget 5 ms per turn for inference (compute target minus search), safety factor 1.5, 24 value nodes per turn, every seat of a match running at once. P is the policy for every opponent team with every head (an upper bound); V is the value network per search node. Times are the worst p99 over every turn of every concurrent seat.

## Tiers

| tier | hidden widths | P MACs / opponent team | V MACs / node | header per model | all 11 models | FUSION_MAX_WIDTH |
|---|---|---:|---:|---:|---:|---:|
| XL | mover 128; squad 512,128; act 128; target 32; value 1024,64 | 273,920 | 120,896 | 2524 KiB | 27.1 MiB | 1024 |
| XXL | mover 256; squad 1024,256; act 256; target 32; value 2048,128 | 678,912 | 372,864 | 6657 KiB | 71.5 MiB | 2048 |
| 3XL | mover 512; squad 2048,512; act 512; target 64; value 4096,256 | 1,882,112 | 1,270,016 | 19602 KiB | 210.6 MiB | 4096 |
| 4XL | mover 1024; squad 4096,1024; act 1024; target 64; value 8192,512 | 5,861,376 | 4,637,184 | 62610 KiB | 672.6 MiB | 8192 |

## Per team count

| teams | map | seats | tier | P p99 / max ms | V per node us | V per turn p99 ms | needed ×1.5 ms | fits | V nodes that fit |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| 15 | 49 | 1 | XL | 0.380 / 0.607 | 3.32 | 0.096 | 0.71 | yes | 889 |
| 15 | 49 | 1 | XXL | 0.826 / 0.873 | 8.24 | 0.291 | 1.68 | yes | 304 |
| 15 | 49 | 1 | 3XL | 1.271 / 1.618 | 24.07 | 0.681 | 2.93 | yes | 85 |
| 15 | 49 | 1 | 4XL | 3.426 / 4.393 | 107.12 | 3.060 | 9.73 | NO | 0 |

## Verdict

- 15 teams: largest tier that fits: **3XL**

**Recommended ceiling for every team count: 3XL.**

Read with care:

- The policy figure includes every head at every legal move; the joint-v2 adapter needs the heads only at sampled moves, so real cost is lower. Search time is not included: the budget is what search leaves.
- Larger networks only help if training data supports them; this table bounds cost, not benefit.
- A tier above FUSION_MAX_WIDTH 256 must be trained and exported with `--max-width` and the bot built with `-DFUSION_MAX_WIDTH`; the exported header refuses to compile otherwise.
- The judge's source-size limit is unknown; the last tier columns give the header text it would add.

## Streams

- 15 teams: n15-s1-000000.json.gz (test bots, real map and engine)

## Kernels: scalar loop vs SIMD

SIMD path on this CPU: **avx2**. Same integers, same results (kernel_check active avx2 paths scalar sse2 avx2 checks 3139 mismatches 0). Speedup = scalar time / SIMD time.

| teams | tier | P p99 scalar / SIMD ms | P speedup | V node scalar / SIMD us | V speedup | needed ×safety scalar / SIMD ms |
|---:|---|---:|---:|---:|---:|---:|
| 15 | XL | 0.440 / 0.380 | 1.16× | 4.92 / 3.32 | 1.48× | 0.87 / 0.71 |
| 15 | XXL | 0.856 / 0.826 | 1.04× | 12.60 / 8.24 | 1.53× | 1.87 / 1.68 |
| 15 | 3XL | 1.626 / 1.271 | 1.28× | 39.38 / 24.07 | 1.64× | 4.09 / 2.93 |
| 15 | 4XL | 4.404 / 3.426 | 1.29× | 151.76 / 107.12 | 1.42× | 12.77 / 9.73 |
