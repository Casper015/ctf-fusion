# Capacity benchmark: DESKTOP-G60OJUD

2026-10-08T16:06:22

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
| 15 | 49 | 16 | XL | 0.720 / 0.993 | 5.15 | 0.201 | 1.38 | yes | 507 |
| 15 | 49 | 16 | XXL | 1.298 / 1.497 | 13.45 | 0.527 | 2.74 | yes | 151 |
| 15 | 49 | 16 | 3XL | 8.264 / 12.506 | 63.82 | 7.093 | 23.04 | NO | 0 |
| 15 | 49 | 16 | 4XL | 45.789 / 53.544 | 2462.27 | 81.879 | 191.50 | NO | 0 |

## Verdict

- 15 teams: largest tier that fits: **XXL**

**Recommended ceiling for every team count: XXL.**

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
| 15 | XL | 0.837 / 0.720 | 1.16× | 7.56 / 5.15 | 1.47× | 1.72 / 1.38 |
| 15 | XXL | 1.706 / 1.298 | 1.31× | 21.76 / 13.45 | 1.62× | 3.88 / 2.74 |
| 15 | 3XL | 6.006 / 8.264 | 0.73× | 83.52 / 63.82 | 1.31× | 15.51 / 23.04 |
| 15 | 4XL | 42.501 / 45.789 | 0.93× | 1918.42 / 2462.27 | 0.78× | 189.25 / 191.50 |
