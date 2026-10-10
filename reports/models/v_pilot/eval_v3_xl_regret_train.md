# Value evaluation: train split

Rows 390216 from 230 maps. Probabilities are calibrated (sigmoid(beta z)). Intervals are 95%, clustered by map; an interval that includes zero is "not shown". Lower loss alone does not establish stronger decisions.

| Run | Family | Hidden | Params | Log loss | Brier | ECE | Pair acc. | Regret | Epochs (best) | Seconds | Peak GB |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| V3_XL | v3 | [1024, 64] | 177411 | 0.4740 | 0.0374 | 0.0078 | 0.509 | 0.0665 | 13 (9) | 7.8 | 0.229 |

## By phase (turns)

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 1-100 | 95910 | 0.5492 | 0.0646 | 0.0102 |
| V3_XL | 101-200 | 98392 | 0.4920 | 0.0437 | 0.0057 |
| V3_XL | 201-300 | 98045 | 0.4498 | 0.0287 | 0.0070 |
| V3_XL | 301-400 | 97869 | 0.4049 | 0.0124 | 0.0124 |

## By team count

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | 2 | 27945 | 0.3090 | 0.0797 | 0.0150 |
| V3_XL | 3 | 30718 | 0.4342 | 0.0646 | 0.0177 |
| V3_XL | 4 | 24049 | 0.3786 | 0.0418 | 0.0238 |
| V3_XL | 5 | 28268 | 0.4255 | 0.0315 | 0.0125 |
| V3_XL | 6 | 32390 | 0.5422 | 0.0414 | 0.0242 |
| V3_XL | 7 | 23402 | 0.5663 | 0.0348 | 0.0084 |
| V3_XL | 8 | 23625 | 0.4800 | 0.0311 | 0.0310 |
| V3_XL | 9 | 25785 | 0.5425 | 0.0290 | 0.0199 |
| V3_XL | 10 | 34753 | 0.5248 | 0.0305 | 0.0119 |
| V3_XL | 11 | 30571 | 0.5069 | 0.0272 | 0.0242 |
| V3_XL | 12 | 28992 | 0.5131 | 0.0304 | 0.0137 |
| V3_XL | 13 | 27581 | 0.4598 | 0.0259 | 0.0149 |
| V3_XL | 14 | 24231 | 0.4698 | 0.0217 | 0.0344 |
| V3_XL | 15 | 27906 | 0.4641 | 0.0277 | 0.0266 |

## Subsets

| Run | Group | Rows | Log loss | Brier | ECE |
|---|---|---:|---:|---:|---:|
| V3_XL | carrier | 344660 | 0.4794 | 0.0343 | 0.0068 |
| V3_XL | drop | 144902 | 0.4863 | 0.0300 | 0.0069 |
| V3_XL | interception | 33491 | 0.4750 | 0.0384 | 0.0080 |
| V3_XL | close | 265798 | 0.5403 | 0.0434 | 0.0053 |

## Decision regret

| Run | Roots | Resolved / uncertain pairs | Regret (V choice) | On-policy | Uniform |
|---|---:|---|---:|---:|---:|
| V3_XL | 118 | 253 / 455 | 0.0665 [0.0454, 0.0877] | 0.0892 | 0.0864 |

## Learning curves

- **V3_XL** (val outcome loss by epoch): 1: 0.4747, 2: 0.4722, 3: 0.4683, 4: 0.4694, 5: 0.4686, 6: 0.4689, 7: 0.4690, 8: 0.4681, 9: 0.4675, 10: 0.4676, 11: 0.4675, 12: 0.4676, 13: 0.4682
