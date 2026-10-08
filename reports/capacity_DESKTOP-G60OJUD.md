# Capacity benchmark: DESKTOP-G60OJUD

2026-10-08T14:30:56

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
| 2 | 23 | 16 | S | 0.032 / 0.108 | 1.66 | 0.077 | 0.16 | yes | 1,983 |
| 2 | 23 | 16 | M | 0.049 / 0.297 | 4.11 | 0.176 | 0.34 | yes | 798 |
| 2 | 23 | 16 | L | 0.087 / 0.237 | 12.78 | 0.521 | 0.91 | yes | 254 |
| 2 | 23 | 16 | XL | 0.170 / 0.306 | 25.56 | 1.027 | 1.80 | yes | 123 |
| 2 | 23 | 16 | XXL | 0.406 / 0.547 | 85.05 | 3.176 | 5.37 | NO | 34 |
| 3 | 31 | 16 | S | 0.064 / 0.166 | 1.73 | 0.071 | 0.20 | yes | 1,891 |
| 3 | 31 | 16 | M | 0.108 / 0.242 | 4.26 | 0.171 | 0.42 | yes | 756 |
| 3 | 31 | 16 | L | 0.167 / 0.316 | 12.72 | 0.500 | 1.00 | yes | 248 |
| 3 | 31 | 16 | XL | 0.345 / 1.427 | 25.78 | 1.007 | 2.03 | yes | 115 |
| 3 | 31 | 16 | XXL | 0.768 / 1.499 | 85.47 | 3.172 | 5.91 | NO | 30 |
| 4 | 31 | 16 | S | 0.086 / 0.443 | 1.74 | 0.073 | 0.24 | yes | 1,870 |
| 4 | 31 | 16 | M | 0.141 / 0.498 | 3.96 | 0.168 | 0.46 | yes | 805 |
| 4 | 31 | 16 | L | 0.248 / 0.437 | 12.65 | 0.517 | 1.15 | yes | 243 |
| 4 | 31 | 16 | XL | 0.482 / 0.668 | 24.33 | 1.001 | 2.22 | yes | 117 |
| 4 | 31 | 16 | XXL | 1.104 / 1.564 | 82.46 | 3.172 | 6.41 | NO | 27 |
| 5 | 31 | 16 | S | 0.138 / 0.330 | 2.27 | 0.093 | 0.35 | yes | 1,410 |
| 5 | 31 | 16 | M | 0.196 / 0.359 | 4.71 | 0.190 | 0.58 | yes | 665 |
| 5 | 31 | 16 | L | 0.325 / 0.460 | 13.84 | 0.538 | 1.29 | yes | 217 |
| 5 | 31 | 16 | XL | 0.671 / 1.484 | 27.28 | 1.073 | 2.62 | yes | 97 |
| 5 | 31 | 16 | XXL | 1.450 / 1.683 | 88.59 | 3.295 | 7.12 | NO | 21 |
| 6 | 41 | 16 | S | 0.151 / 0.285 | 1.92 | 0.076 | 0.34 | yes | 1,657 |
| 6 | 41 | 16 | M | 0.242 / 0.401 | 4.20 | 0.169 | 0.62 | yes | 735 |
| 6 | 41 | 16 | L | 0.397 / 0.770 | 12.63 | 0.492 | 1.33 | yes | 232 |
| 6 | 41 | 16 | XL | 0.798 / 1.117 | 24.47 | 0.952 | 2.62 | yes | 103 |
| 6 | 41 | 16 | XXL | 1.809 / 2.042 | 83.04 | 3.095 | 7.36 | NO | 18 |
| 7 | 41 | 16 | S | 0.179 / 0.550 | 2.16 | 0.084 | 0.39 | yes | 1,459 |
| 7 | 41 | 16 | M | 0.287 / 0.576 | 4.74 | 0.180 | 0.70 | yes | 643 |
| 7 | 41 | 16 | L | 0.487 / 1.153 | 13.67 | 0.517 | 1.51 | yes | 208 |
| 7 | 41 | 16 | XL | 0.950 / 1.178 | 26.41 | 1.012 | 2.94 | yes | 90 |
| 7 | 41 | 16 | XXL | 2.153 / 2.492 | 86.69 | 3.187 | 8.01 | NO | 13 |
| 8 | 41 | 16 | S | 0.207 / 0.513 | 2.32 | 0.088 | 0.44 | yes | 1,346 |
| 8 | 41 | 16 | M | 0.332 / 0.703 | 4.94 | 0.196 | 0.79 | yes | 607 |
| 8 | 41 | 16 | L | 0.555 / 0.894 | 14.46 | 0.538 | 1.64 | yes | 192 |
| 8 | 41 | 16 | XL | 1.103 / 1.305 | 28.06 | 1.056 | 3.24 | yes | 79 |
| 8 | 41 | 16 | XXL | 2.505 / 2.820 | 89.55 | 3.275 | 8.67 | NO | 9 |
| 9 | 41 | 16 | S | 0.237 / 0.371 | 2.36 | 0.088 | 0.49 | yes | 1,309 |
| 9 | 41 | 16 | M | 0.393 / 0.646 | 5.01 | 0.184 | 0.87 | yes | 587 |
| 9 | 41 | 16 | L | 0.630 / 1.029 | 14.13 | 0.525 | 1.73 | yes | 191 |
| 9 | 41 | 16 | XL | 1.278 / 1.579 | 27.50 | 1.029 | 3.46 | yes | 74 |
| 9 | 41 | 16 | XXL | 2.848 / 3.160 | 89.01 | 3.327 | 9.26 | NO | 5 |
| 10 | 41 | 16 | S | 0.274 / 0.649 | 2.40 | 0.089 | 0.54 | yes | 1,276 |
| 10 | 41 | 16 | M | 0.440 / 0.671 | 4.72 | 0.177 | 0.93 | yes | 612 |
| 10 | 41 | 16 | L | 0.714 / 0.966 | 13.51 | 0.507 | 1.83 | yes | 193 |
| 10 | 41 | 16 | XL | 1.432 / 1.669 | 26.18 | 0.989 | 3.63 | yes | 72 |
| 10 | 41 | 16 | XXL | 3.196 / 3.449 | 86.38 | 3.140 | 9.50 | NO | 1 |
| 11 | 49 | 16 | S | 0.293 / 0.580 | 2.30 | 0.088 | 0.57 | yes | 1,319 |
| 11 | 49 | 16 | M | 0.476 / 0.770 | 4.61 | 0.172 | 0.97 | yes | 620 |
| 11 | 49 | 16 | L | 0.794 / 0.970 | 13.02 | 0.480 | 1.91 | yes | 195 |
| 11 | 49 | 16 | XL | 1.574 / 1.767 | 25.30 | 0.934 | 3.76 | yes | 69 |
| 11 | 49 | 16 | XXL | 3.551 / 4.016 | 83.92 | 3.035 | 9.88 | NO | 0 |
| 12 | 49 | 16 | S | 0.321 / 0.615 | 2.26 | 0.090 | 0.62 | yes | 1,334 |
| 12 | 49 | 16 | M | 0.524 / 0.976 | 4.45 | 0.163 | 1.03 | yes | 631 |
| 12 | 49 | 16 | L | 0.869 / 1.273 | 12.67 | 0.471 | 2.01 | yes | 194 |
| 12 | 49 | 16 | XL | 1.718 / 2.112 | 24.03 | 0.898 | 3.92 | yes | 67 |
| 12 | 49 | 16 | XXL | 3.902 / 4.735 | 81.48 | 2.984 | 10.33 | NO | 0 |
| 13 | 49 | 16 | S | 0.349 / 0.599 | 2.38 | 0.094 | 0.66 | yes | 1,256 |
| 13 | 49 | 16 | M | 0.572 / 0.943 | 4.69 | 0.178 | 1.12 | yes | 588 |
| 13 | 49 | 16 | L | 0.956 / 1.134 | 12.98 | 0.469 | 2.14 | yes | 183 |
| 13 | 49 | 16 | XL | 1.865 / 2.059 | 24.53 | 0.902 | 4.15 | yes | 59 |
| 13 | 49 | 16 | XXL | 4.207 / 4.492 | 82.72 | 2.977 | 10.78 | NO | 0 |
| 14 | 49 | 16 | S | 0.375 / 0.494 | 2.48 | 0.093 | 0.70 | yes | 1,194 |
| 14 | 49 | 16 | M | 0.608 / 0.843 | 4.81 | 0.179 | 1.18 | yes | 566 |
| 14 | 49 | 16 | L | 1.035 / 1.445 | 13.23 | 0.484 | 2.28 | yes | 173 |
| 14 | 49 | 16 | XL | 2.022 / 2.301 | 25.47 | 0.912 | 4.40 | yes | 51 |
| 14 | 49 | 16 | XXL | 4.539 / 4.984 | 84.65 | 3.023 | 11.34 | NO | 0 |
| 15 | 49 | 16 | S | 0.410 / 1.821 | 2.62 | 0.099 | 0.76 | yes | 1,116 |
| 15 | 49 | 16 | M | 0.662 / 0.940 | 4.98 | 0.178 | 1.26 | yes | 536 |
| 15 | 49 | 16 | L | 1.095 / 1.461 | 13.99 | 0.502 | 2.40 | yes | 160 |
| 15 | 49 | 16 | XL | 2.204 / 2.619 | 26.85 | 0.948 | 4.73 | yes | 42 |
| 15 | 49 | 16 | XXL | 4.937 / 5.276 | 87.06 | 3.078 | 12.02 | NO | 0 |

## Verdict

- 2 teams: largest tier that fits: **XL**
- 3 teams: largest tier that fits: **XL**
- 4 teams: largest tier that fits: **XL**
- 5 teams: largest tier that fits: **XL**
- 6 teams: largest tier that fits: **XL**
- 7 teams: largest tier that fits: **XL**
- 8 teams: largest tier that fits: **XL**
- 9 teams: largest tier that fits: **XL**
- 10 teams: largest tier that fits: **XL**
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
