# Capacity benchmark: DESKTOP-G60OJUD

2026-10-08T14:35:17

Budget 5 ms per turn for inference (compute target minus search), safety factor 1.5, 24 value nodes per turn, every seat of a match running at once. P is the policy for every opponent team with every head (an upper bound); V is the value network per search node. Times are the worst p99 over every turn of every concurrent seat.

## Tiers

| tier | hidden widths | P MACs / opponent team | V MACs / node | header per model | all 11 models | FUSION_MAX_WIDTH |
|---|---|---:|---:|---:|---:|---:|
| S | mover 16; squad 64,32; act 16; target 8; value 128,16 | 30,096 | 8,976 | 209 KiB | 2.2 MiB | 256 |
| M | mover 32; squad 128,64; act 32; target 16; value 256,32 | 64,288 | 22,048 | 470 KiB | 5.0 MiB | 256 |
| L | mover 64; squad 256,64; act 64; target 16; value 512,64 | 120,576 | 60,480 | 993 KiB | 10.7 MiB | 512 |
| XL | mover 128; squad 512,128; act 128; target 32; value 1024,64 | 273,920 | 120,896 | 2192 KiB | 23.5 MiB | 1024 |
| XXL | mover 256; squad 1024,256; act 256; target 32; value 2048,128 | 678,912 | 372,864 | 5992 KiB | 64.4 MiB | 2048 |

## Per team count

| teams | map | seats | tier | P p99 / max ms | V per node us | V per turn p99 ms | needed ×1.5 ms | fits | V nodes that fit |
|---:|---:|---:|---|---:|---:|---:|---:|---|---:|
| 2 | 23 | 1 | S | 0.019 / 0.032 | 1.12 | 0.040 | 0.09 | yes | 2,948 |
| 2 | 23 | 1 | M | 0.031 / 0.032 | 2.78 | 0.101 | 0.20 | yes | 1,187 |
| 2 | 23 | 1 | L | 0.056 / 0.061 | 8.32 | 0.314 | 0.55 | yes | 393 |
| 2 | 23 | 1 | XL | 0.113 / 0.117 | 16.68 | 0.638 | 1.13 | yes | 193 |
| 2 | 23 | 1 | XXL | 0.223 / 0.343 | 54.09 | 1.696 | 2.88 | yes | 57 |
| 3 | 31 | 1 | S | 0.033 / 0.048 | 1.15 | 0.038 | 0.11 | yes | 2,872 |
| 3 | 31 | 1 | M | 0.062 / 0.100 | 2.74 | 0.095 | 0.24 | yes | 1,196 |
| 3 | 31 | 1 | L | 0.078 / 0.087 | 8.28 | 0.251 | 0.49 | yes | 393 |
| 3 | 31 | 1 | XL | 0.154 / 0.220 | 16.20 | 0.512 | 1.00 | yes | 196 |
| 3 | 31 | 1 | XXL | 0.331 / 0.468 | 53.09 | 1.543 | 2.81 | yes | 56 |
| 4 | 31 | 1 | S | 0.049 / 0.062 | 1.19 | 0.040 | 0.13 | yes | 2,766 |
| 4 | 31 | 1 | M | 0.076 / 0.172 | 2.73 | 0.089 | 0.25 | yes | 1,193 |
| 4 | 31 | 1 | L | 0.117 / 0.132 | 8.21 | 0.254 | 0.56 | yes | 391 |
| 4 | 31 | 1 | XL | 0.225 / 0.256 | 15.99 | 0.499 | 1.09 | yes | 194 |
| 4 | 31 | 1 | XXL | 0.645 / 0.904 | 53.32 | 1.860 | 3.76 | yes | 50 |
| 5 | 31 | 1 | S | 0.068 / 0.083 | 1.49 | 0.049 | 0.18 | yes | 2,192 |
| 5 | 31 | 1 | M | 0.109 / 0.156 | 3.22 | 0.101 | 0.32 | yes | 1,000 |
| 5 | 31 | 1 | L | 0.161 / 0.168 | 9.18 | 0.278 | 0.66 | yes | 345 |
| 5 | 31 | 1 | XL | 0.299 / 0.328 | 17.78 | 0.532 | 1.25 | yes | 170 |
| 5 | 31 | 1 | XXL | 0.654 / 0.791 | 56.66 | 1.597 | 3.38 | yes | 47 |
| 6 | 41 | 1 | S | 0.080 / 0.093 | 1.30 | 0.040 | 0.18 | yes | 2,498 |
| 6 | 41 | 1 | M | 0.125 / 0.198 | 2.83 | 0.085 | 0.32 | yes | 1,134 |
| 6 | 41 | 1 | L | 0.267 / 0.286 | 8.32 | 0.284 | 0.83 | yes | 368 |
| 6 | 41 | 1 | XL | 0.374 / 0.546 | 16.31 | 0.474 | 1.27 | yes | 181 |
| 6 | 41 | 1 | XXL | 1.039 / 1.143 | 53.62 | 1.641 | 4.02 | yes | 42 |
| 7 | 41 | 1 | S | 0.096 / 0.108 | 1.49 | 0.045 | 0.21 | yes | 2,166 |
| 7 | 41 | 1 | M | 0.199 / 0.277 | 3.18 | 0.128 | 0.49 | yes | 986 |
| 7 | 41 | 1 | L | 0.306 / 0.318 | 9.16 | 0.315 | 0.93 | yes | 330 |
| 7 | 41 | 1 | XL | 0.606 / 0.692 | 17.93 | 0.593 | 1.80 | yes | 152 |
| 7 | 41 | 1 | XXL | 1.001 / 1.404 | 55.75 | 1.605 | 3.91 | yes | 41 |
| 8 | 41 | 1 | S | 0.115 / 0.136 | 1.63 | 0.049 | 0.25 | yes | 1,970 |
| 8 | 41 | 1 | M | 0.243 / 0.253 | 3.51 | 0.125 | 0.55 | yes | 880 |
| 8 | 41 | 1 | L | 0.282 / 0.364 | 9.57 | 0.286 | 0.85 | yes | 318 |
| 8 | 41 | 1 | XL | 0.753 / 0.866 | 18.81 | 0.635 | 2.08 | yes | 137 |
| 8 | 41 | 1 | XXL | 1.122 / 1.258 | 57.80 | 1.572 | 4.04 | yes | 38 |
| 9 | 41 | 1 | S | 0.136 / 0.180 | 1.70 | 0.050 | 0.28 | yes | 1,881 |
| 9 | 41 | 1 | M | 0.203 / 0.210 | 3.40 | 0.098 | 0.45 | yes | 921 |
| 9 | 41 | 1 | L | 0.392 / 0.438 | 9.36 | 0.326 | 1.08 | yes | 314 |
| 9 | 41 | 1 | XL | 0.748 / 0.841 | 18.13 | 0.625 | 2.06 | yes | 142 |
| 9 | 41 | 1 | XXL | 1.562 / 1.814 | 56.89 | 1.708 | 4.91 | yes | 31 |
| 10 | 41 | 1 | S | 0.155 / 0.168 | 1.68 | 0.051 | 0.31 | yes | 1,894 |
| 10 | 41 | 1 | M | 0.235 / 0.259 | 3.30 | 0.096 | 0.50 | yes | 940 |
| 10 | 41 | 1 | L | 0.353 / 0.449 | 8.95 | 0.252 | 0.91 | yes | 333 |
| 10 | 41 | 1 | XL | 0.673 / 0.987 | 17.22 | 0.486 | 1.74 | yes | 154 |
| 10 | 41 | 1 | XXL | 1.409 / 1.918 | 55.12 | 1.496 | 4.36 | yes | 34 |
| 11 | 49 | 1 | S | 0.182 / 0.231 | 1.66 | 0.056 | 0.36 | yes | 1,896 |
| 11 | 49 | 1 | M | 0.336 / 0.358 | 3.30 | 0.121 | 0.69 | yes | 907 |
| 11 | 49 | 1 | L | 0.480 / 0.595 | 8.62 | 0.299 | 1.17 | yes | 331 |
| 11 | 49 | 1 | XL | 0.734 / 0.823 | 16.43 | 0.467 | 1.80 | yes | 158 |
| 11 | 49 | 1 | XXL | 1.800 / 2.347 | 53.66 | 1.583 | 5.07 | NO | 28 |
| 12 | 49 | 1 | S | 0.180 / 0.225 | 1.63 | 0.051 | 0.35 | yes | 1,932 |
| 12 | 49 | 1 | M | 0.296 / 0.434 | 3.13 | 0.100 | 0.59 | yes | 970 |
| 12 | 49 | 1 | L | 0.420 / 0.617 | 8.24 | 0.227 | 0.97 | yes | 353 |
| 12 | 49 | 1 | XL | 0.828 / 0.981 | 15.67 | 0.442 | 1.91 | yes | 159 |
| 12 | 49 | 1 | XXL | 2.313 / 2.600 | 51.88 | 1.775 | 6.13 | NO | 19 |
| 13 | 49 | 1 | S | 0.200 / 0.223 | 1.73 | 0.051 | 0.38 | yes | 1,805 |
| 13 | 49 | 1 | M | 0.295 / 0.376 | 3.22 | 0.093 | 0.58 | yes | 943 |
| 13 | 49 | 1 | L | 0.458 / 0.796 | 8.49 | 0.244 | 1.05 | yes | 338 |
| 13 | 49 | 1 | XL | 1.165 / 1.259 | 16.18 | 0.541 | 2.56 | yes | 133 |
| 13 | 49 | 1 | XXL | 2.033 / 2.640 | 52.61 | 1.483 | 5.27 | NO | 24 |
| 14 | 49 | 1 | S | 0.237 / 0.316 | 1.79 | 0.058 | 0.44 | yes | 1,729 |
| 14 | 49 | 1 | M | 0.327 / 0.397 | 3.31 | 0.092 | 0.63 | yes | 907 |
| 14 | 49 | 1 | L | 0.510 / 0.850 | 8.79 | 0.240 | 1.12 | yes | 321 |
| 14 | 49 | 1 | XL | 0.986 / 1.354 | 16.64 | 0.459 | 2.17 | yes | 141 |
| 14 | 49 | 1 | XXL | 2.034 / 2.134 | 53.64 | 1.455 | 5.23 | NO | 24 |
| 15 | 49 | 1 | S | 0.259 / 0.270 | 1.87 | 0.062 | 0.48 | yes | 1,641 |
| 15 | 49 | 1 | M | 0.353 / 0.478 | 3.44 | 0.098 | 0.68 | yes | 867 |
| 15 | 49 | 1 | L | 0.685 / 0.745 | 9.10 | 0.308 | 1.49 | yes | 290 |
| 15 | 49 | 1 | XL | 1.336 / 1.480 | 17.21 | 0.582 | 2.88 | yes | 116 |
| 15 | 49 | 1 | XXL | 2.255 / 3.145 | 54.80 | 1.512 | 5.65 | NO | 19 |

## Verdict

- 2 teams: largest tier that fits: **XXL**
- 3 teams: largest tier that fits: **XXL**
- 4 teams: largest tier that fits: **XXL**
- 5 teams: largest tier that fits: **XXL**
- 6 teams: largest tier that fits: **XXL**
- 7 teams: largest tier that fits: **XXL**
- 8 teams: largest tier that fits: **XXL**
- 9 teams: largest tier that fits: **XXL**
- 10 teams: largest tier that fits: **XXL**
- 11 teams: largest tier that fits: **XL**
- 12 teams: largest tier that fits: **XL**
- 13 teams: largest tier that fits: **XL**
- 14 teams: largest tier that fits: **XL**
- 15 teams: largest tier that fits: **XL**

**Recommended ceiling for every team count: XL.**

Read with care:

- The policy figure includes every head at every legal move; the joint-v2 adapter needs the heads only at sampled moves, so real cost is lower. Search time is not included: the budget is what search leaves.
- Larger networks only help if training data supports them; this table bounds cost, not benefit.
- A tier above FUSION_MAX_WIDTH 256 must be trained and exported with `--max-width` and the bot built with `-DFUSION_MAX_WIDTH`; the exported header refuses to compile otherwise.
- The judge's source-size limit is unknown; the last tier columns give the header text it would add.

## Streams

- 2 teams: n02-s1-000000.json.gz (test bots, real map and engine)
- 3 teams: n03-s1-000000.json.gz (test bots, real map and engine)
- 4 teams: n04-s1-000000.json.gz (test bots, real map and engine)
- 5 teams: n05-s1-000000.json.gz (test bots, real map and engine)
- 6 teams: n06-s1-000000.json.gz (test bots, real map and engine)
- 7 teams: n07-s1-000000.json.gz (test bots, real map and engine)
- 8 teams: n08-s1-000000.json.gz (test bots, real map and engine)
- 9 teams: n09-s1-000000.json.gz (test bots, real map and engine)
- 10 teams: n10-s1-000000.json.gz (test bots, real map and engine)
- 11 teams: n11-s1-000000.json.gz (test bots, real map and engine)
- 12 teams: n12-s1-000000.json.gz (test bots, real map and engine)
- 13 teams: n13-s1-000000.json.gz (test bots, real map and engine)
- 14 teams: n14-s1-000000.json.gz (test bots, real map and engine)
- 15 teams: n15-s1-000000.json.gz (test bots, real map and engine)
